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
//
// This implementation workers as follows:
//   - Each worker is a classical tensorflow parameter server with the custom
//     ops defined in this directory.
//   - The manager connects to each of the workers and create two remote ops
//     (i.e. ops controlled by the manager, but running on the
//     workers):YggdrasilDistributeRunTask and YggdrasilDistributeStopWorker.
//   - YggdrasilDistributeRunTask execute a task on a worker (called when the
//     user schedule a task), YggdrasilDistributeStopWorker stops the worker
//     (called when the user call "Done").
//   - Each worker also connects to each other workers and create a
//     YggdrasilDistributeRunInterWorkerTask remote OP. This ops runs a task
//     (like YggdrasilDistributeRunTask). However, unlike
//     YggdrasilDistributeRunTask, YggdrasilDistributeRunInterWorkerTask does
//     not have all the worker initialization logic.
//
// Troubleshooting and common errors:
//   - Depending on the environment variables and flags, when a client connects
//     to a server, a server is created automatically on the client side. When
//     multiple connection are made (one for each worker), only the first
//     connection will start this client-side-server (protected by a local
//     mutex). However, this local mutex is not shared in between the C++ and
//     Python APIs. Therefore, in some case, you will receive "address already
//     used" fatal error messages. The solution is set to the env variable
//     "TF_GRPC_REUSE_PORT=1" before any connection is made (i.e. add
//     os.environ['TF_GRPC_REUSE_PORT'] = '1' in your python code).
//
#ifndef TENSORFLOW_DECISION_FORESTS_TENSORFLOW_DISTRIBUTE_TF_DISTRIBUTION_H_
#define TENSORFLOW_DECISION_FORESTS_TENSORFLOW_DISTRIBUTE_TF_DISTRIBUTION_H_

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribute/core.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.pb.h"
#include "yggdrasil_decision_forests/utils/distribute/utils.h"

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
                           const std::vector<std::string>& worker_addresses,
                           const std::vector<std::string>& worker_resource_ids,
                           int parallel_execution_per_worker,
                           std::unique_ptr<RemoteConnection>* connection);

  // Run a task.
  utils::StatusOr<Blob> RunTask(const Blob& input);

  // Stop the remote worker.
  absl::Status StopWorker(bool kill_worker_manager);

 private:
  tf::Status status;
  std::unique_ptr<tf::ClientSession> session;
  std::unique_ptr<tf::Scope> root;
  std::string target;
  utils::concurrency::SharedMutex session_mutex;

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

  absl::Status SetParallelExecutionPerWorker(int num) override;

 private:
  struct Worker {
    int worker_idx;
    std::string address;

    ThreadVector process_local_queries;
    ThreadVector process_global_queries;

    // Async query to execute specific to this worker.
    utils::concurrency::Channel<Blob> async_pending_queries_;
    std::unique_ptr<internal::RemoteConnection> connection;
    std::string resource_uid;

    void StartThreads(int parallel_execution_per_worker,
                      TfDistributionManager* manager);
  };

  absl::Status Initialize(const proto::Config& config,
                          absl::string_view worker_name, Blob welcome_blob,
                          int parallel_execution_per_worker) override;

  absl::Status InitializeWorkers(const proto::Config& config,
                                 absl::string_view worker_name,
                                 Blob welcome_blob,
                                 int parallel_execution_per_worker);

  // Creates the name of the resource container that will hold the connection
  // data on the worker side. The resource ids are used for intra-worker
  // communication and deduplication of TF ops.
  std::vector<std::string> CreateWorkerResourceIds(
      absl::string_view worker_name, int num_workers);

  void WorkerRun(Blob blob, Worker* worker);

  // Thread loop to process the global and worker-specific queries.
  void ProcessGlobalQueries(Worker* worker);
  void ProcessLocalQueries(Worker* worker);

  void JoinWorkers();

  int verbosity_;
  std::vector<std::unique_ptr<Worker>> workers_;

  // Async query to execute by any worker.
  utils::concurrency::Channel<Blob> async_pending_queries_;

  // Async answers that should be returned, indexed by task.
  utils::concurrency::Channel<utils::StatusOr<Blob>> async_pending_answers_;

  // Idx of the next worker to receive a job if the worker idx is not
  // specified by the user.
  std::atomic<int> next_auto_worker_idx_ = {0};

  // Check if "Done" was called. "Done" will be called as the object
  // destruction if it was not called manually before.
  bool done_was_called_ = false;
};

REGISTER_Distribution_Manager(TfDistributionManager,
                              TfDistributionManager::kKey);

}  // namespace distribute
}  // namespace yggdrasil_decision_forests

#endif  // TENSORFLOW_DECISION_FORESTS_TENSORFLOW_DISTRIBUTE_TF_DISTRIBUTION_H_
