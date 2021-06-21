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


#include "absl/status/status.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribute/core.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.pb.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/tensorflow.h"

namespace yggdrasil_decision_forests {
namespace distribute {

// Name of the "tf resource group" containing models loaded in memory.
constexpr char kResourceContainer[] = "yggdrasil_decision_forests_distribute";

namespace tf = ::tensorflow;
using OpKernel = tf::OpKernel;
using OpKernelConstruction = tf::OpKernelConstruction;
using OpKernelContext = tf::OpKernelContext;
using TensorShape = tf::TensorShape;
using Tensor = tf::Tensor;

// Worker manager containing the worker instance.
class WorkerResource : public tf::ResourceBase {
 public:
  std::string DebugString() const override { return "WorkerResource"; }

  utils::StatusOr<Blob> RunTask(Blob blob) {
    absl::ReaderMutexLock l(&mu_);
    return worker_->RunRequest(blob);
  }

  absl::Status ReadyWorker(const std::string& welcome_blob,
                           const std::string& worker_name,
                           const int worker_idx) {
    absl::WriterMutexLock l(&mu_);
    ASSIGN_OR_RETURN(worker_, AbstractWorkerRegisterer::Create(worker_name));
    RETURN_IF_ERROR(InternalInitializeWorker(worker_idx, worker_.get()));
    RETURN_IF_ERROR(worker_->Setup(welcome_blob));
    return absl::OkStatus();
  }

  absl::Status Done() {
    absl::WriterMutexLock l(&mu_);
    if (worker_) {
      RETURN_IF_ERROR(worker_->Done());
      worker_.reset();
    }
    return absl::OkStatus();
  }

 private:
  absl::Mutex mu_;
  std::unique_ptr<AbstractWorker> worker_ ABSL_GUARDED_BY(mu_);
};

class YggdrasilDistributeRunTask : public OpKernel {
 public:
  explicit YggdrasilDistributeRunTask(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("welcome_blob", &welcome_blob_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("resource_uid", &resource_uid_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("worker_name", &worker_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("worker_idx", &worker_idx_));
  }

  ~YggdrasilDistributeRunTask() override {
    if (worker_resource_) {
      worker_resource_->Unref();
      worker_resource_ = nullptr;
    }
  }

  void Compute(OpKernelContext* ctx) override {
    absl::MutexLock l(&mu_);
    if (!worker_resource_) {
      OP_REQUIRES_OK(ctx, CreateWorkerResource(ctx));
    }

    const Tensor* input_blob_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input_blob", &input_blob_tensor));
    const auto input_blob = input_blob_tensor->flat<tf::tstring>()(0);

    Tensor* output_blob_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("output_blob", TensorShape({}),
                                             &output_blob_tensor));

    auto result_or = worker_resource_->RunTask(input_blob);
    if (!result_or.ok()) {
      OP_REQUIRES_OK(ctx, utils::FromUtilStatus(result_or.status()));
    } else {
      output_blob_tensor->flat<tf::tstring>()(0) = std::move(result_or).value();
    }
  }

 private:
  tf::Status CreateWorkerResource(OpKernelContext* ctx)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    TF_RETURN_IF_ERROR(ctx->resource_manager()->LookupOrCreate<WorkerResource>(
        kResourceContainer, resource_uid_, &worker_resource_,
        [&](WorkerResource** resource) -> tensorflow::Status {
          *resource = new WorkerResource();
          return tensorflow::Status::OK();
        }));

    TF_RETURN_IF_ERROR(utils::FromUtilStatus(worker_resource_->ReadyWorker(
        welcome_blob_, worker_name_, worker_idx_)));

    return tf::Status::OK();
  }

  int worker_idx_;
  std::string welcome_blob_;
  std::string worker_name_;
  std::string resource_uid_;
  absl::Mutex mu_;
  WorkerResource* worker_resource_ ABSL_GUARDED_BY(mu_) = nullptr;
};

REGISTER_KERNEL_BUILDER(
    Name("YggdrasilDistributeRunTask").Device(tf::DEVICE_CPU),
    YggdrasilDistributeRunTask);

class YggdrasilDistributeStopWorker : public OpKernel {
 public:
  explicit YggdrasilDistributeStopWorker(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("resource_uid", &resource_uid_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* kill_worker_manager_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->input("kill_worker_manager", &kill_worker_manager_tensor));
    const auto kill_worker_manager =
        kill_worker_manager_tensor->flat<bool>()(0);

    WorkerResource* worker_resource_;
    if (ctx->resource_manager()
            ->Lookup<WorkerResource>(kResourceContainer, resource_uid_,
                                     &worker_resource_)
            .ok()) {
      OP_REQUIRES_OK(ctx, utils::FromUtilStatus(worker_resource_->Done()));
      worker_resource_->Unref();

      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Delete<WorkerResource>(
                              kResourceContainer, resource_uid_));
    }

    if (kill_worker_manager) {
      LOG(INFO) << "Killing process because kill_worker_manager=true";

      killing_thread_ = absl::make_unique<utils::concurrency::Thread>([]() {
        // Enough time for the GRPC to send back the ACK.
        absl::SleepFor(absl::Seconds(3));
        std::exit(0);
      });
    }
  }

 private:
  std::string resource_uid_;

  // Thread that will kill the server instance.
  std::unique_ptr<utils::concurrency::Thread> killing_thread_;
};

REGISTER_KERNEL_BUILDER(
    Name("YggdrasilDistributeStopWorker").Device(tf::DEVICE_CPU),
    YggdrasilDistributeStopWorker);

}  // namespace distribute
}  // namespace yggdrasil_decision_forests
