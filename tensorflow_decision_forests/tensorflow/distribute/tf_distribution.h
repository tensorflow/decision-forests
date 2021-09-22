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

// Implementation of the Yggdrasil Distribute interface to use a TF Distribution
// strategy.

#ifndef TENSORFLOW_DECISION_FORESTS_TENSORFLOW_DISTRIBUTE_TF_DISTRIBUTION_H_
#define TENSORFLOW_DECISION_FORESTS_TENSORFLOW_DISTRIBUTE_TF_DISTRIBUTION_H_

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribute/core.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.pb.h"

namespace yggdrasil_decision_forests {
namespace distribute {

namespace tf = ::tensorflow;

namespace internal {

// Connection to a remote TF session ran by a remote TF Server. It can run the
// "RunTask" and "StopWorker" custom c++ op remotely.
// This class is instantiated in the manager job for each remote communication
// "channel" -- one per worker.
class RemoteConnection {
 public:
  RemoteConnection() {}

  static tf::Status Create(const std::string& target, const Blob& welcome_blob,
                           absl::string_view worker_name, int worker_idx,
                           const int num_workers,
                           std::unique_ptr<RemoteConnection>* connection);

  // Run a task.
  utils::StatusOr<Blob> RunTask(Blob&& input);

  // Stop the remote worker.
  absl::Status StopWorker(bool kill_worker_manager);

 private:
  tf::Status status;
  std::unique_ptr<tf::ClientSession> session;
  std::unique_ptr<tf::Scope> root;

  // Graph.
  std::unique_ptr<tf::ops::Placeholder> run_task_input;
  tf::NodeBuilder::NodeOut run_task_input_node;
  tf::Node* run_task_node;
  tf::Operation run_task_op;
  tf::Output run_task_output;

  std::unique_ptr<tf::ops::Placeholder> stop_worker_input;
  tf::NodeBuilder::NodeOut stop_worker_input_node;
  tf::Node* stop_worker_node;
  tf::Operation stop_worker_op;
};

utils::StatusOr<std::vector<std::string>> JsonConfigToWorkers(
    absl::string_view json);

}  // namespace internal

class TfDistributionManager : public AbstractManager {
 public:
  static constexpr char kKey[] = "TF_DIST";

  virtual ~TfDistributionManager() {
    if (!done_was_called_) {
      LOG(WARNING) << "Calling Done in distribution manager destructor";
      CHECK_OK(Done({}));
    }
  }

  utils::StatusOr<Blob> BlockingRequest(Blob blob, int worker_idx) override;

  absl::Status AsynchronousRequest(Blob blob, int worker_idx) override;

  utils::StatusOr<Blob> NextAsynchronousAnswer() override;

  int NumWorkers() override;

  absl::Status Done(absl::optional<bool> kill_worker_manager) override;

  utils::StatusOr<int> NumWorkersInConfiguration(
      const proto::Config& config) const override;

 private:
  struct Worker {
    int worker_idx;
    std::string address;
    // Threads running executing the global and worker specific requests.
    std::unique_ptr<utils::concurrency::Thread> main_thread_local_pool;
    std::unique_ptr<utils::concurrency::Thread> main_thread_global_pool;
    // Async query to execute specific to this worker.
    utils::concurrency::Channel<Blob> async_pending_queries_;
    std::unique_ptr<internal::RemoteConnection> connection;
    std::string resource_uid;
  };

  absl::Status Initialize(const proto::Config& config,
                          absl::string_view worker_name, Blob welcome_blob,
                          int parallel_execution_per_worker) override;

  absl::Status InitializeWorkers(const proto::Config& config,
                                 absl::string_view worker_name,
                                 Blob welcome_blob);

  void WorkerRun(Blob blob, Worker* worker);

  void WorkerMainLocalPool(Worker* worker);
  void WorkerMainGlobalPool(Worker* worker);

  void JoinWorkers();

  int verbosity_;
  std::vector<std::unique_ptr<Worker>> workers_;

  // Async query to execute by any worker.
  utils::concurrency::Channel<Blob> async_pending_queries_;

  // Async answers that should be returned, indexed by task.
  utils::concurrency::Channel<utils::StatusOr<Blob>> async_pending_answers_;

  // Idx of the next worker to receive a job if the worker idx is not specified
  // by the user.
  std::atomic<int> next_auto_worker_idx_ = {0};

  // Check if "Done" was called. "Done" will be called as the object destruction
  // if it was not called manually before.
  bool done_was_called_ = false;
};

REGISTER_Distribution_Manager(TfDistributionManager,
                              TfDistributionManager::kKey);

}  // namespace distribute
}  // namespace yggdrasil_decision_forests

#endif  // TENSORFLOW_DECISION_FORESTS_TENSORFLOW_DISTRIBUTE_TF_DISTRIBUTION_H_
