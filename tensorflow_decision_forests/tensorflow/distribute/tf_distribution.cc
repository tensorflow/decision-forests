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

// Manager code.

#include "tensorflow_decision_forests/tensorflow/distribute/tf_distribution.h"

#include <random>

#include "third_party/rapidjson/include/rapidjson/document.h"
#include "third_party/rapidjson/include/rapidjson/reader.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow_decision_forests/tensorflow/distribute/tf_distribution.pb.h"
#include "tensorflow_decision_forests/tensorflow/distribute/tf_distribution_common.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/concurrency_channel.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"
#include "yggdrasil_decision_forests/utils/tensorflow.h"
#include "yggdrasil_decision_forests/utils/uid.h"

namespace yggdrasil_decision_forests {
namespace distribute {
namespace tf = ::tensorflow;

constexpr char TfDistributionManager::kKey[];

utils::StatusOr<int> TfDistributionManager::NumWorkersInConfiguration(
    const proto::Config& config) const {
  const auto& imp_config = config.GetExtension(proto::tf_distribution);

  switch (imp_config.worker_address_case()) {
    case proto::TfDistribution::kAddresses:
      return imp_config.addresses().addresses_size();
    case proto::TfDistribution::kEnvironmentVariable: {
      const char* tf_config = std::getenv("TF_CONFIG");
      if (tf_config == nullptr) {
        return absl::InvalidArgumentError("TF_CONFIG not defined");
      }
      ASSIGN_OR_RETURN(const auto worker_addresses,
                       internal::JsonConfigToWorkers(tf_config));
      return worker_addresses.size();
    }
    default:
      return absl::UnimplementedError("Unknown worker address type");
  }
}

std::vector<std::string> TfDistributionManager::CreateWorkerResourceIds(
    absl::string_view worker_name, const int num_workers) {
  std::vector<std::string> resource_ids;
  resource_ids.reserve(num_workers);
  for (int worker_idx = 0; worker_idx < num_workers; worker_idx++) {
    resource_ids.push_back(absl::Substitute("$0_$1_$2", worker_name, worker_idx,
                                            utils::GenUniqueId()));
  }
  return resource_ids;
}

absl::Status TfDistributionManager::InitializeWorkers(
    const proto::Config& config, const absl::string_view worker_name,
    Blob welcome_blob, const int parallel_execution_per_worker) {
  const auto& imp_config = config.GetExtension(proto::tf_distribution);

  std::vector<std::string> worker_addresses;
  switch (imp_config.worker_address_case()) {
    case proto::TfDistribution::kAddresses:
      for (const auto& address : imp_config.addresses().addresses()) {
        worker_addresses.push_back(address);
      }
      break;
    case proto::TfDistribution::kEnvironmentVariable: {
      const char* tf_config = std::getenv("TF_CONFIG");
      if (tf_config == nullptr) {
        return absl::InvalidArgumentError("TF_CONFIG not defined");
      }
      ASSIGN_OR_RETURN(worker_addresses,
                       internal::JsonConfigToWorkers(tf_config));
    } break;

    default:
      return absl::UnimplementedError("Unknown worker address type");
  }

  if (worker_addresses.empty()) {
    return absl::InvalidArgumentError("There should be at least one worker");
  }

  if (verbosity_ >= 1) {
    LOG(INFO) << "Start manager with " << worker_addresses.size()
              << " workers.";
  }

  const auto worker_resource_ids =
      CreateWorkerResourceIds(worker_name, worker_addresses.size());

  for (int worker_idx = 0; worker_idx < worker_addresses.size(); worker_idx++) {
    auto worker = absl::make_unique<Worker>();
    worker->worker_idx = worker_idx;
    worker->address = worker_addresses[worker_idx];

    if (verbosity_ >= 1) {
      LOG(INFO) << "Connect to worker #" << worker_idx << " at address "
                << worker_addresses[worker_idx];
    }

    CHECK_OK(utils::ToUtilStatus(internal::RemoteConnection::Create(
        worker_addresses[worker_idx], welcome_blob, worker_name, worker_idx,
        worker_addresses, worker_resource_ids, parallel_execution_per_worker,
        &worker->connection)));

    worker->StartThreads(parallel_execution_per_worker, this);
    workers_.push_back(std::move(worker));
  }
  return absl::OkStatus();
}

void TfDistributionManager::WorkerRun(Blob blob, Worker* worker) {
  auto result = worker->connection->RunTask(blob);
  async_pending_answers_.Push(std::move(result));
}

void TfDistributionManager::ProcessLocalQueries(Worker* worker) {
  while (true) {
    auto pending_blob_or = worker->async_pending_queries_.Pop();
    if (!pending_blob_or.has_value()) {
      break;
    }
    WorkerRun(std::move(pending_blob_or.value()), worker);
  }
}

void TfDistributionManager::ProcessGlobalQueries(Worker* worker) {
  while (true) {
    auto pending_blob_or = async_pending_queries_.Pop();
    if (!pending_blob_or.has_value()) {
      break;
    }
    WorkerRun(std::move(pending_blob_or.value()), worker);
  }
}

utils::StatusOr<Blob> TfDistributionManager::BlockingRequest(Blob blob,
                                                             int worker_idx) {
  if (verbosity_ >= 2) {
    LOG(INFO) << "Sending blocking request with " << blob.size() << " bytes";
  }
  if (worker_idx < 0) {
    worker_idx = next_auto_worker_idx_.fetch_add(1) % workers_.size();
  }
  auto* worker = workers_[worker_idx].get();
  return worker->connection->RunTask(blob);
}

absl::Status TfDistributionManager::AsynchronousRequest(Blob blob,
                                                        int worker_idx) {
  if (verbosity_ >= 2) {
    LOG(INFO) << "Sending asynchronous request with " << blob.size()
              << " bytes";
  }
  if (worker_idx < 0) {
    async_pending_queries_.Push(std::move(blob));
  } else {
    workers_[worker_idx]->async_pending_queries_.Push(std::move(blob));
  }
  return absl::OkStatus();
}

utils::StatusOr<Blob> TfDistributionManager::NextAsynchronousAnswer() {
  if (verbosity_ >= 2) {
    LOG(INFO) << "Wait for next asynchronous result";
  }
  auto answer_or = async_pending_answers_.Pop();
  if (!answer_or.has_value()) {
    return absl::OutOfRangeError("No more results available");
  }
  if (verbosity_ >= 2) {
    LOG(INFO) << "Receive asynchronous request with "
              << answer_or.value().value().size() << " bytes";
  }
  return std::move(answer_or.value());
}

int TfDistributionManager::NumWorkers() { return workers_.size(); }

absl::Status TfDistributionManager::SetParallelExecutionPerWorker(int num) {
  if (verbosity_) {
    LOG(INFO) << "Change the number of parallel execution per worker";
  }

  // Close the query channels.
  async_pending_queries_.Close();
  for (auto& worker : workers_) {
    worker->async_pending_queries_.Close();
  }

  // Wait for the threads to join
  JoinWorkers();

  // Re-open the channels and restart the threads.
  async_pending_queries_.Reopen();
  for (auto& worker : workers_) {
    worker->async_pending_queries_.Reopen();
    worker->StartThreads(num, this);
  }
  return absl::OkStatus();
}

void TfDistributionManager::Worker::StartThreads(
    int parallel_execution_per_worker, TfDistributionManager* manager) {
  process_local_queries.Start(parallel_execution_per_worker, [this, manager]() {
    manager->ProcessLocalQueries(this);
  });

  process_global_queries.Start(
      parallel_execution_per_worker,
      [this, manager]() { manager->ProcessGlobalQueries(this); });
}

absl::Status TfDistributionManager::Done(
    absl::optional<bool> kill_worker_manager) {
  if (verbosity_ >= 1) {
    LOG(INFO) << "Shutdown manager";
  }
  if (done_was_called_) {
    LOG(WARNING) << "Calling done twice";
    return absl::OkStatus();
  }
  done_was_called_ = true;
  async_pending_queries_.Close();
  async_pending_answers_.Close();

  for (auto& worker : workers_) {
    worker->async_pending_queries_.Close();
  }

  if (verbosity_ >= 1) {
    LOG(INFO) << "Joining workers";
  }
  JoinWorkers();
  if (verbosity_ >= 1) {
    LOG(INFO) << "Joining workers done";
  }

  if (verbosity_ >= 1) {
    LOG(INFO) << "Killing workers";
  }
  // TODO: Run in parallel.
  for (auto& worker : workers_) {
    auto worker_shutdown =
        worker->connection->StopWorker(kill_worker_manager.value_or(false));
    if (!worker_shutdown.ok()) {
      // It is not a big deal if the worker crashes during shutdown.
      LOG(WARNING) << "Worker error during shutdown. "
                   << worker_shutdown.message();
    }
  }

  workers_.clear();

  if (verbosity_ >= 1) {
    LOG(INFO) << "Manager has been shutdown";
  }

  return absl::OkStatus();
}

void TfDistributionManager::JoinWorkers() {
  for (auto& worker : workers_) {
    worker->process_local_queries.JoinAndClear();
    worker->process_global_queries.JoinAndClear();
  }
}

absl::Status TfDistributionManager::Initialize(
    const proto::Config& config, const absl::string_view worker_name,
    Blob welcome_blob, int parallel_execution_per_worker) {
  verbosity_ = config.verbosity();

  if (verbosity_ >= 2) {
    LOG(INFO) << "Initialize manager with " << welcome_blob.size()
              << " bytes welcome blob";
  }

  RETURN_IF_ERROR(InitializeWorkers(config, worker_name,
                                    std::move(welcome_blob),
                                    parallel_execution_per_worker));
  return absl::OkStatus();
}

namespace internal {

tf::Status RemoteConnection::Create(
    const std::string& target, const Blob& welcome_blob,
    const absl::string_view worker_name, const int worker_idx,
    const std::vector<std::string>& worker_addresses,
    const std::vector<std::string>& worker_resource_ids,
    const int parallel_execution_per_worker,
    std::unique_ptr<RemoteConnection>* connection) {
  auto conn = absl::make_unique<RemoteConnection>();
  conn->root = absl::make_unique<tf::Scope>(tf::Scope::NewRootScope());

  // Run task op
  {
    conn->run_task_input =
        absl::make_unique<tf::ops::Placeholder>(*conn->root, tf::DT_STRING);
    if (!conn->root->ok()) return conn->root->status();

    conn->run_task_input_node =
        ::tensorflow::ops::AsNodeOut(*conn->root, *conn->run_task_input);
    if (!conn->root->ok()) return conn->root->status();

    const auto unique_name =
        conn->root->GetUniqueNameForOp("YggdrasilDistributeRunTask");
    auto builder =
        ::tensorflow::NodeBuilder(unique_name, "YggdrasilDistributeRunTask")
            .Input(conn->run_task_input_node)
            .Attr("welcome_blob", welcome_blob)
            .Attr("worker_name", std::string(worker_name))
            .Attr("worker_idx", worker_idx)
            .Attr("worker_addresses", worker_addresses)
            .Attr("worker_resource_ids", worker_resource_ids)
            .Attr("parallel_execution_per_worker",
                  parallel_execution_per_worker)
            .Attr("resource_uid", worker_resource_ids[worker_idx]);
    conn->root->UpdateBuilder(&builder);
    conn->root->UpdateStatus(
        builder.Finalize(conn->root->graph(), &conn->run_task_node));
    if (!conn->root->ok()) return conn->root->status();

    conn->run_task_op = tf::Operation(conn->run_task_node);
    conn->run_task_output = tf::Output(conn->run_task_node, 0);
  }

  // Stop worker op
  {
    conn->stop_worker_input =
        absl::make_unique<tf::ops::Placeholder>(*conn->root, tf::DT_BOOL);
    if (!conn->root->ok()) return conn->root->status();

    conn->stop_worker_input_node =
        ::tensorflow::ops::AsNodeOut(*conn->root, *conn->stop_worker_input);
    if (!conn->root->ok()) return conn->root->status();

    const auto unique_name =
        conn->root->GetUniqueNameForOp("YggdrasilDistributeStopWorker");
    auto builder =
        ::tensorflow::NodeBuilder(unique_name, "YggdrasilDistributeStopWorker")
            .Input(conn->stop_worker_input_node)
            .Attr("resource_uid", worker_resource_ids[worker_idx]);
    conn->root->UpdateBuilder(&builder);
    conn->root->UpdateStatus(
        builder.Finalize(conn->root->graph(), &conn->stop_worker_node));
    if (!conn->root->ok()) return conn->root->status();

    conn->stop_worker_op = tf::Operation(conn->stop_worker_node);
  }

  conn->target = target;
  conn->session = absl::make_unique<tf::ClientSession>(*conn->root, target);
  *connection = std::move(conn);
  return (*connection)->root->status();
}

absl::Status RemoteConnection::StopWorker(const bool kill_worker_manager) {
  tf::ClientSession::FeedType feeds;
  feeds.emplace(tf::Output(*stop_worker_input),
                tf::Input::Initializer(kill_worker_manager));

  tf::RunOptions options;
  if (kill_worker_manager) {
    options.set_timeout_in_ms(2000);
  }
  std::vector<tf::Tensor> outputs;
  return utils::ToUtilStatus(
      session->Run(options, feeds, {}, {stop_worker_op}, &outputs, nullptr));
}

utils::StatusOr<Blob> RemoteConnection::RunTask(const Blob& input) {
  tf::ClientSession::FeedType feeds;
  feeds.emplace(tf::Output(*run_task_input), tf::Input::Initializer(input));
  std::vector<tf::Tensor> outputs;

  int num_re_emitting = 0;
  while (true) {
    absl::Status status;
    {
      utils::concurrency::ReaderMutexLock lock(&session_mutex);
      status =
          utils::ToUtilStatus(session->Run(feeds, {run_task_output}, &outputs));
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

    utils::concurrency::WriterMutexLock lock(&session_mutex);
    session = absl::make_unique<tf::ClientSession>(*root, target);
  }
}

utils::StatusOr<std::vector<std::string>> JsonConfigToWorkers(
    const absl::string_view json) {
  std::vector<std::string> workers;

  rapidjson::Document dom;
  dom.Parse(std::string(json).c_str());

  if (!dom.IsObject()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid TF_CONFIG. No object found. json: ", json));
  }

  std::string rpc_layer = "grpc";  // Default to GRPC.
  auto rpc_layer_it = dom.GetObject().FindMember("rpc_layer");
  if (rpc_layer_it != dom.GetObject().MemberEnd()) {
    if (!rpc_layer_it->value.IsString()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid TF_CONFIG. rpc_layer is not a string. json: ", json));
    }
    rpc_layer = rpc_layer_it->value.GetString();
  }

  auto cluster_it = dom.GetObject().FindMember("cluster");
  if (cluster_it == dom.GetObject().MemberEnd()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid TF_CONFIG. No cluster found. json: ", json));
  }
  if (!cluster_it->value.IsObject()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid TF_CONFIG. cluster is not an object. json: ", json));
  }
  auto worker_it = cluster_it->value.FindMember("worker");
  if (worker_it == cluster_it->value.MemberEnd()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid TF_CONFIG. No worker found in cluster. json: ", json));
  }
  if (!worker_it->value.IsArray()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid TF_CONFIG. worker is not an array. json: ", json));
  }
  auto json_workers = worker_it->value.GetArray();
  for (auto worker_item_it = json_workers.begin();
       worker_item_it != json_workers.end(); worker_item_it++) {
    if (!worker_item_it->IsString()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid TF_CONFIG. worker item is not a string. json: ", json));
    }
    workers.push_back(
        absl::StrCat(rpc_layer, "://", worker_item_it->GetString()));
  }

  return workers;
}

}  // namespace internal

}  // namespace distribute
}  // namespace yggdrasil_decision_forests
