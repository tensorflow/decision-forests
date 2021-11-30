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
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow_decision_forests/tensorflow/distribute/tf_distribution_common.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribute/core.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.pb.h"
#include "yggdrasil_decision_forests/utils/distribute/utils.h"
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
  WorkerResource() : hook_(this) {}

  std::string DebugString() const override { return "WorkerResource"; }

  utils::StatusOr<Blob> RunTask(Blob blob) {
    absl::ReaderMutexLock lock(&mu_);
    if (!worker_) {
      return absl::InternalError("Worker no set");
    }
    auto result = worker_->RunRequest(blob);
    if (!result.ok()) {
      LOG(WARNING) << "RunTask failed with error: "
                   << result.status().ToString();
    }
    return result;
  }

  absl::Status ReadyWorker(const std::string& welcome_blob,
                           const std::string& worker_name, const int worker_idx,
                           const std::vector<std::string>& worker_addresses,
                           const std::vector<std::string>& worker_resource_ids,
                           const int parallel_execution_per_worker) {
    absl::WriterMutexLock lock(&mu_);
    ASSIGN_OR_RETURN(worker_, AbstractWorkerRegisterer::Create(worker_name));
    RETURN_IF_ERROR(InternalInitializeWorker(
        worker_idx, worker_addresses.size(), worker_.get(), &hook_));
    RETURN_IF_ERROR(worker_->Setup(welcome_blob));
    RETURN_IF_ERROR(InitializerInterWorkerCommunication(
        worker_addresses, worker_resource_ids, parallel_execution_per_worker));
    return absl::OkStatus();
  }

  absl::Status Done() {
    absl::WriterMutexLock lock(&mu_);
    if (worker_) {
      RETURN_IF_ERROR(worker_->Done());
      worker_.reset();
    }

    FinalizeIntraWorkerCommunication();
    return absl::OkStatus();
  }

 protected:
  // Implementation of worker->worker request.
  absl::Status AsynchronousRequestToOtherWorker(
      Blob blob, int target_worker_idx, AbstractWorker* emitter_worker) {
    intra_worker_communication_.pending_queries.Push(
        std::make_pair(target_worker_idx, std::move(blob)));
    return absl::OkStatus();
  }

  // Implementation of the worker->worker async reply.
  utils::StatusOr<Blob> NextAsynchronousAnswerFromOtherWorker(
      AbstractWorker* emitter_worker) {
    auto answer = intra_worker_communication_.pending_answers.Pop();
    if (!answer.has_value()) {
      return absl::OutOfRangeError("No more results available");
    }
    return std::move(answer.value());
  }

 private:
  class WorkerHook : public AbstractWorkerHook {
   public:
    WorkerHook(WorkerResource* resource) : resource_(resource) {}

    absl::Status AsynchronousRequestToOtherWorker(
        Blob blob, int target_worker_idx,
        AbstractWorker* emitter_worker) override {
      return resource_->AsynchronousRequestToOtherWorker(
          std::move(blob), target_worker_idx, emitter_worker);
    }

    utils::StatusOr<Blob> NextAsynchronousAnswerFromOtherWorker(
        AbstractWorker* emitter_worker) override {
      return resource_->NextAsynchronousAnswerFromOtherWorker(emitter_worker);
    }

   private:
    WorkerResource* resource_;
  };

  // Fields related to the inter worker communication.
  struct InterWorkerCommunication {
    // List of target worker index and data emitted by this worker.
    utils::concurrency::Channel<std::pair<int, Blob>> pending_queries;

    // Answers to this worker queries.
    utils::concurrency::Channel<utils::StatusOr<Blob>> pending_answers;

    // Thread emitting and receiving intra-workers requests/answers.
    ThreadVector threads;

    struct OtherWorkers {
      absl::Mutex mutex;
      std::string socket_address;
      std::string resource_id;

      std::unique_ptr<tf::ClientSession> session;  // ABSL_GUARDED_BY(mutex);

      tf::Status status;
      std::unique_ptr<tf::Scope> root;

      // Graph.
      std::unique_ptr<tf::ops::Placeholder> run_task_input;
      tf::NodeBuilder::NodeOut run_task_input_node;
      tf::Node* run_task_node;
      tf::Operation run_task_op;
      tf::Output run_task_output;
    };

    // Communication channel to other workers for intra worker communication.
    std::vector<std::unique_ptr<OtherWorkers>> other_workers;
  };

  // Initialize the connection and thread for the inter worker communication.
  // This method should be called before any inter worker communication.
  absl::Status InitializerInterWorkerCommunication(
      const std::vector<std::string>& worker_addresses,
      const std::vector<std::string>& worker_resource_ids,
      const int num_threads) {
    if (worker_addresses.size() != worker_resource_ids.size()) {
      return absl::InternalError(
          "Non matching worker_addresses and worker_resource_ids");
    }

    intra_worker_communication_.other_workers.resize(worker_addresses.size());
    for (int worker_idx = 0; worker_idx < worker_addresses.size();
         worker_idx++) {
      auto& worker = intra_worker_communication_.other_workers[worker_idx];
      worker = absl::make_unique<InterWorkerCommunication::OtherWorkers>();
      worker->resource_id = worker_resource_ids[worker_idx];
      worker->socket_address = worker_addresses[worker_idx];
    }
    intra_worker_communication_.threads.Start(
        num_threads, [&]() { ProcessInterWorkerCommunication(); });
    return absl::OkStatus();
  }

  // Finalize the current worker communication.
  // No more inter worker communication should be done after this call, except
  // for "InitializerInterWorkerCommunication" to re-initialize it.
  void FinalizeIntraWorkerCommunication() {
    intra_worker_communication_.pending_answers.Close();
    intra_worker_communication_.pending_queries.Close();
    intra_worker_communication_.threads.JoinAndClear();
  }

  // Ensures that the communication with another worker is ready.
  tf::Status EnsureIntraWorkerStubIsReady(
      InterWorkerCommunication::OtherWorkers* worker) {
    absl::MutexLock lock(&worker->mutex);
    CHECK(worker);

    if (worker->session) {
      // Already connected.
      return tf::Status();
    }

    // Connection.
    worker->root = absl::make_unique<tf::Scope>(tf::Scope::NewRootScope());

    // Run task op
    {
      worker->run_task_input =
          absl::make_unique<tf::ops::Placeholder>(*worker->root, tf::DT_STRING);
      if (!worker->root->ok()) {
        return worker->root->status();
      }

      worker->run_task_input_node =
          ::tensorflow::ops::AsNodeOut(*worker->root, *worker->run_task_input);
      if (!worker->root->ok()) {
        return worker->root->status();
      }

      const auto unique_name = worker->root->GetUniqueNameForOp(
          "YggdrasilDistributeRunInterWorkerTask");
      auto builder = ::tensorflow::NodeBuilder(
                         unique_name, "YggdrasilDistributeRunInterWorkerTask")
                         .Input(worker->run_task_input_node)
                         .Attr("resource_uid", worker->resource_id);
      worker->root->UpdateBuilder(&builder);
      worker->root->UpdateStatus(
          builder.Finalize(worker->root->graph(), &worker->run_task_node));
      if (!worker->root->ok()) {
        return worker->root->status();
      }

      worker->run_task_op = tf::Operation(worker->run_task_node);
      worker->run_task_output = tf::Output(worker->run_task_node, 0);
    }

    worker->session = absl::make_unique<tf::ClientSession>(
        *worker->root, worker->socket_address);
    return worker->root->status();
  }

  // Blocking inter worker request.
  utils::StatusOr<Blob> BlockingInterWorkerRequest(
      Blob blob, InterWorkerCommunication::OtherWorkers* target_worker) {
    RETURN_IF_ERROR(
        utils::ToUtilStatus(EnsureIntraWorkerStubIsReady(target_worker)));

    tf::ClientSession::FeedType feeds;
    feeds.emplace(tf::Output(*target_worker->run_task_input),
                  tf::Input::Initializer(blob));
    std::vector<tf::Tensor> outputs;

    int num_re_emitting = 0;
    while (true) {
      absl::Status status;
      {
        absl::ReaderMutexLock lock(&target_worker->mutex);
        status = utils::ToUtilStatus(target_worker->session->Run(
            feeds, {target_worker->run_task_output}, &outputs));
      }

      if (status.ok()) {
        return outputs[0].flat<tf::tstring>()(0);
      }

      if (IsPermanentWorkerError(status)) {
        LOG(WARNING) << "Session call failed with permanent error: "
                     << status.ToString();
        return status;
      }
      LOG(WARNING) << "Session call failed with non-permanent error: "
                   << status.ToString();

      if (num_re_emitting > 10) {
        LOG(WARNING) << "Too much re-tries. Returning last error status";
        return status;
      }

      num_re_emitting++;
      LOG(WARNING) << "Re-emitting request (num_re_emitting:" << num_re_emitting
                   << ")";
      absl::SleepFor(absl::Seconds(10));

      absl::WriterMutexLock lock(&target_worker->mutex);
      target_worker->session = absl::make_unique<tf::ClientSession>(
          *target_worker->root, target_worker->socket_address);
    }
  }

  // Loop for a thread processing inter worker requests.
  void ProcessInterWorkerCommunication() {
    while (true) {
      auto pending_blob_or = intra_worker_communication_.pending_queries.Pop();
      if (!pending_blob_or.has_value()) {
        break;
      }
      const auto target_worker_idx = pending_blob_or.value().first;
      auto* target_worker =
          intra_worker_communication_.other_workers[target_worker_idx].get();

      auto answer = BlockingInterWorkerRequest(
          std::move(pending_blob_or).value().second, target_worker);

      if (!answer.ok()) {
        LOG(WARNING) << "Inter worker communication failed with error: "
                     << answer.status().message();
      }
      intra_worker_communication_.pending_answers.Push(std::move(answer));
    }
  }

  absl::Mutex mu_;
  std::unique_ptr<AbstractWorker> worker_ ABSL_GUARDED_BY(mu_);
  WorkerHook hook_;
  InterWorkerCommunication intra_worker_communication_;
};

class YggdrasilDistributeRunTask : public OpKernel {
 public:
  explicit YggdrasilDistributeRunTask(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("welcome_blob", &welcome_blob_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("resource_uid", &resource_uid_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("worker_name", &worker_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("worker_idx", &worker_idx_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("worker_addresses", &worker_addresses_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("worker_resource_ids", &worker_resource_ids_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("parallel_execution_per_worker",
                                     &parallel_execution_per_worker_));
  }

  ~YggdrasilDistributeRunTask() override {
    if (worker_resource_) {
      worker_resource_->Unref();
      worker_resource_ = nullptr;
    }
  }

  void Compute(OpKernelContext* ctx) override {
    {
      absl::MutexLock lock(&mutex_);
      if (!worker_resource_) {
        OP_REQUIRES_OK(ctx, CreateWorkerResource(ctx));
      }
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
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    TF_RETURN_IF_ERROR(ctx->resource_manager()->LookupOrCreate<WorkerResource>(
        kResourceContainer, resource_uid_, &worker_resource_,
        [&](WorkerResource** resource) -> tensorflow::Status {
          *resource = new WorkerResource();
          return tensorflow::Status::OK();
        }));

    TF_RETURN_IF_ERROR(utils::FromUtilStatus(worker_resource_->ReadyWorker(
        welcome_blob_, worker_name_, worker_idx_, worker_addresses_,
        worker_resource_ids_, parallel_execution_per_worker_)));

    return tf::Status::OK();
  }

  int worker_idx_;
  std::vector<std::string> worker_addresses_;
  std::vector<std::string> worker_resource_ids_;
  int parallel_execution_per_worker_;
  std::string welcome_blob_;
  std::string worker_name_;
  std::string resource_uid_;
  absl::Mutex mutex_;
  WorkerResource* worker_resource_ = nullptr;
};

REGISTER_KERNEL_BUILDER(
    Name("YggdrasilDistributeRunTask").Device(tf::DEVICE_CPU),
    YggdrasilDistributeRunTask);

class YggdrasilDistributeRunInterWorkerTask : public OpKernel {
 public:
  explicit YggdrasilDistributeRunInterWorkerTask(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("resource_uid", &resource_uid_));
  }

  ~YggdrasilDistributeRunInterWorkerTask() override {
    if (worker_resource_) {
      worker_resource_->Unref();
      worker_resource_ = nullptr;
    }
  }

  void Compute(OpKernelContext* ctx) override {
    absl::MutexLock l(&mutex_);
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
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup<WorkerResource>(
        kResourceContainer, resource_uid_, &worker_resource_));
    return tf::Status::OK();
  }

  std::string resource_uid_;
  absl::Mutex mutex_;
  WorkerResource* worker_resource_ ABSL_GUARDED_BY(mutex_) = nullptr;
};

REGISTER_KERNEL_BUILDER(
    Name("YggdrasilDistributeRunInterWorkerTask").Device(tf::DEVICE_CPU),
    YggdrasilDistributeRunInterWorkerTask);

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
