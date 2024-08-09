/*
 * Copyright 2021 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>

#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "tensorflow_decision_forests/tensorflow/ops/training/kernel.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc_manager.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc_worker.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace tensorflow_decision_forests {
namespace ops {

namespace {
namespace tf = ::tensorflow;
namespace ydf = ::yggdrasil_decision_forests;
namespace utils = ydf::utils;
}  // namespace

// Name of the TF resource container containing the resources
// "RunningProcessResource".
constexpr char kTFContainer[] = "ydf_grpc";

// Holds the running GRPC servers data.
class YDFGRPCServerResource : public ::tensorflow::ResourceBase {
 public:
  YDFGRPCServerResource() = default;

  virtual ~YDFGRPCServerResource() override { StopServer(); };

  std::string DebugString() const override { return ""; }

  // Port of the running server. -1 if the server is not running.
  int32_t port() const {
    if (!server_) {
      return -1;
    }
    return server_->port;
  }

  // Blocking function running the worker.
  void ThreadMain() {
    ydf::distribute::grpc_worker::WaitForGRPCWorkerToShutdown(server_.get());
    LOG(INFO) << "YDF GRPC Worker stopped";
  }

  // Starts the GRPC in a new thread.
  //
  // Args:
  //   force_ydf_port: Port to use. If -1, select automatically a port.
  absl::Status StartServer(int force_ydf_port) {
    if (server_) {
      return absl::InvalidArgumentError("Server already running");
    }
    ASSIGN_OR_RETURN(server_,
                     ydf::distribute::grpc_worker::StartGRPCWorker(
                         /*port=*/(force_ydf_port != -1) ? force_ydf_port : 0,
                         /*use_loas=*/false));
    LOG(INFO) << "GRPC worker started on port " << server_->port;
    thread_ = absl::make_unique<utils::concurrency::Thread>(
        [this]() { return ThreadMain(); });
    return absl::OkStatus();
  }

  void StopServer() {
    LOG(INFO) << "Stop YDF GRPC Worker";
    if (server_) {
      server_->stop_server.Notify();
    }
    if (thread_) {
      thread_->Join();
      thread_.reset();
    }
    server_.reset();
  }

 private:
  std::unique_ptr<ydf::distribute::grpc_worker::GRPCWorkerServer> server_;
  std::unique_ptr<utils::concurrency::Thread> thread_;
};

class SimpleMLCreateYDFGRPCWorker : public tensorflow::OpKernel {
 public:
  explicit SimpleMLCreateYDFGRPCWorker(tf::OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("key", &key_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("force_ydf_port", &force_ydf_port_));
  }

  ~SimpleMLCreateYDFGRPCWorker() override {}

  void Compute(tf::OpKernelContext* ctx) override {
    // Get the running server / create a new server.
    YDFGRPCServerResource* server_resource;
    OP_REQUIRES_OK(
        ctx,
        ctx->resource_manager()->LookupOrCreate<YDFGRPCServerResource, true>(
            kTFContainer, absl::StrCat(key_), &server_resource,
            [&](YDFGRPCServerResource** resource) -> tensorflow::Status {
              *resource = new YDFGRPCServerResource();
              return utils::FromUtilStatus(
                  (*resource)->StartServer(force_ydf_port_));
            }));

    // Returns the server port.
    tf::Tensor* port_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, tf::TensorShape({}), &port_tensor));
    port_tensor->scalar<int32_t>()() = server_resource->port();
  }

 private:
  // Unique identifier of the GRPC server in the TF session.
  int key_;
  int force_ydf_port_;
};

REGISTER_KERNEL_BUILDER(
    Name("SimpleMLCreateYDFGRPCWorker").Device(tf::DEVICE_CPU),
    SimpleMLCreateYDFGRPCWorker);

class SimpleMLUpdateGRPCWorkerAddress : public tensorflow::OpKernel {
 public:
  explicit SimpleMLUpdateGRPCWorkerAddress(tf::OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("key", &key_));
  }

  ~SimpleMLUpdateGRPCWorkerAddress() override {}

  void Compute(tf::OpKernelContext* ctx) override {
    const auto worker_idx = ctx->input(0).scalar<int32_t>()();
    const auto new_address = ctx->input(1).scalar<tf::tstring>()();
    ydf::distribute::UpdateWorkerAddress(key_, worker_idx,
                                         std::string(new_address));
  }

 private:
  // Unique identifier of the GRPC server in the TF session.
  int key_;
};

REGISTER_KERNEL_BUILDER(
    Name("SimpleMLUpdateGRPCWorkerAddress").Device(tf::DEVICE_CPU),
    SimpleMLUpdateGRPCWorkerAddress);

class SimpleMLStopYDFGRPCWorker : public tensorflow::OpKernel {
 public:
  explicit SimpleMLStopYDFGRPCWorker(tf::OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("key", &key_));
  }

  ~SimpleMLStopYDFGRPCWorker() override {}

  void Compute(tf::OpKernelContext* ctx) override {
    // Get and stop the running server, if any.
    YDFGRPCServerResource* server_resource;
    const auto lookup_status = ctx->resource_manager()->Lookup(
        kTFContainer, absl::StrCat(key_), &server_resource);
    if (lookup_status.ok()) {
      server_resource->StopServer();
      server_resource->Unref();
    }
  }

 private:
  // Unique identifier of the GRPC server in the TF session.
  int key_;
};

REGISTER_KERNEL_BUILDER(
    Name("SimpleMLStopYDFGRPCWorker").Device(tf::DEVICE_CPU),
    SimpleMLStopYDFGRPCWorker);

}  // namespace ops
}  // namespace tensorflow_decision_forests
