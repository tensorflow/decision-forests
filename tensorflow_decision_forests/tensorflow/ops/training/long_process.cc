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

// Utility ops to run asynchronous processes.
//
// See "op.cc" for the documentation of the ops.
//
#include "absl/random/random.h"
#include "tensorflow_decision_forests/tensorflow/ops/training/kernel.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace tensorflow_decision_forests {
namespace ops {

namespace {
namespace tf = ::tensorflow;
namespace ydf = ::yggdrasil_decision_forests;
namespace utils = ydf::utils;
}  // namespace

// Number of seconds waited by "GetLongRunningProcessStatus" when checking the
// status of a running process.
constexpr int kLongRunningProcessCheckTimeOut = 10;

// Name of the resource containing containing the resources
// "RunningProcessResource".
constexpr char kProcessContainer[] = "decision_forests_process";

// A long running process.
class RunningProcessResource : public ::tensorflow::ResourceBase {
 public:
  RunningProcessResource() = default;

  virtual ~RunningProcessResource() override {
    if (thread_) {
      thread_->Join();
    }
  };

  std::string DebugString() const override { return ""; }

  // Start a long running process. Can only be called once.
  void Run(std::function<absl::Status()>&& call) {
    utils::concurrency::MutexLock l(&mutex_);
    DCHECK(thread_ == nullptr);
    call_ = std::move(call);
    status_ = LongRunningProcessStatus::kInProgress;
    thread_ = absl::make_unique<utils::concurrency::Thread>([this]() {
      auto status = call_();

      utils::concurrency::MutexLock l(&mutex_);
      if (!status.ok()) {
        status_ = status;
      } else {
        status_ = LongRunningProcessStatus::kSuccess;
      }
      cond_var_.SignalAll();
    });
  }

  // Gets the status of the process. Wait for kLongRunningProcessCheckTimeOut
  // seconds before returning "in progress".
  absl::StatusOr<LongRunningProcessStatus> GetStatus() {
    utils::concurrency::MutexLock l(&mutex_);
    if (status_.ok() && status_.value() == kInProgress) {
      cond_var_.WaitWithTimeout(&mutex_, &l, kLongRunningProcessCheckTimeOut);
    }
    return status_;
  }

 private:
  std::function<absl::Status()> call_;
  utils::concurrency::Mutex mutex_;
  utils::concurrency::CondVar cond_var_;
  absl::StatusOr<LongRunningProcessStatus> status_ GUARDED_BY(mutex_);
  std::unique_ptr<utils::concurrency::Thread> thread_;
};

absl::StatusOr<int32_t> StartLongRunningProcess(
    tf::OpKernelContext* ctx, std::function<absl::Status()>&& call) {
  // Generate a random name of the process.
  absl::BitGen bitgen;
  int32_t process_id = absl::Uniform(bitgen, 0, 0x7FFFFFFF);

  // Start the process.
  auto* process_container = new RunningProcessResource();
  auto status = ctx->resource_manager()->Create(
      kProcessContainer, absl::StrCat(process_id), process_container);
  if (!status.ok()) {
    return utils::ToUtilStatus(status);
  }
  process_container->Run(std::move(call));
  return process_id;
}

absl::StatusOr<LongRunningProcessStatus> GetLongRunningProcessStatus(
    tf::OpKernelContext* ctx, int32_t process_id) {
  // Get the rtf resource containing the running process.
  const auto resource_name = absl::StrCat(process_id);
  RunningProcessResource* process_container = nullptr;
  auto find_container_status =
      ctx->resource_manager()->Lookup<RunningProcessResource, true>(
          kProcessContainer, resource_name, &process_container);
  if (!find_container_status.ok()) {
    return utils::ToUtilStatus(find_container_status);
  }

  auto process_status = process_container->GetStatus();
  process_container->Unref();

  if (!process_status.ok() ||
      process_status.value() == LongRunningProcessStatus::kSuccess) {
    // Release the process container if the run is done (success or failure).
    auto delete_container_status =
        ctx->resource_manager()->Delete<RunningProcessResource>(
            kProcessContainer, resource_name);
    if (!find_container_status.ok()) {
      return utils::ToUtilStatus(delete_container_status);
    }
  }

  // Return the status.
  return process_status;
}

class SimpleMLCheckStatus : public tensorflow::OpKernel {
 public:
  explicit SimpleMLCheckStatus(tf::OpKernelConstruction* ctx) : OpKernel(ctx) {}

  ~SimpleMLCheckStatus() override = default;

  void Compute(tf::OpKernelContext* ctx) override {
    // Get process id.
    const auto process_id = ctx->input(0).scalar<int32_t>()();

    // Check process status.
    auto status_or = GetLongRunningProcessStatus(ctx, process_id);
    if (!status_or.ok()) {
      OP_REQUIRES_OK(ctx, utils::FromUtilStatus(status_or.status()));
    }

    // Output process status.
    tf::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, tf::TensorShape({}), &output_tensor));
    output_tensor->scalar<int32_t>()() = status_or.value();
  }

 private:
};

REGISTER_KERNEL_BUILDER(Name("SimpleMLCheckStatus").Device(tf::DEVICE_CPU),
                        SimpleMLCheckStatus);

}  // namespace ops
}  // namespace tensorflow_decision_forests
