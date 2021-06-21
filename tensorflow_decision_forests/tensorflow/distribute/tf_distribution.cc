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

#include "absl/synchronization/notification.h"
#include "include/rapidjson/document.h"
#include "include/rapidjson/reader.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow_decision_forests/tensorflow/distribute/tf_distribution.pb.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/concurrency_channel.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/tensorflow.h"

namespace yggdrasil_decision_forests {
namespace distribute {
namespace tf = ::tensorflow;

absl::Status TfDistributionManager::InitializeWorkers(
    const proto::Config& config, const absl::string_view worker_name,
    Blob welcome_blob) {
  const auto& imp_config = config.GetExtension(proto::tf_distribution);

  std::vector<std::string> worker_addresses;
  switch (imp_config.worker_address_case()) {
    case proto::TfDistribution::kSocketAddresses:
      for (const auto& address : imp_config.socket_addresses().addresses()) {
        worker_addresses.push_back(
            absl::StrCat("grpc://", address.ip(), ":", address.port()));
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

  if (verbose_) {
    LOG(INFO) << "Start manager with " << worker_addresses.size()
              << " workers.";
  }

  for (int worker_idx = 0; worker_idx < worker_addresses.size(); worker_idx++) {
    auto worker = absl::make_unique<Worker>();
    worker->worker_idx = worker_idx;
    worker->address = worker_addresses[worker_idx];

    if (verbose_) {
      LOG(INFO) << "Connect to worker #" << worker_idx << " to "
                << worker_addresses[worker_idx];
    }

    CHECK_OK(utils::ToUtilStatus(internal::RemoteConnection::Create(
        worker_addresses[worker_idx], welcome_blob, worker_name, worker_idx,
        &worker->connection)));

    worker->main_thread_local_pool =
        absl::make_unique<utils::concurrency::Thread>(
            [this, worker = worker.get()]() { WorkerMainLocalPool(worker); });

    worker->main_thread_global_pool =
        absl::make_unique<utils::concurrency::Thread>(
            [this, worker = worker.get()]() { WorkerMainGlobalPool(worker); });
    workers_.push_back(std::move(worker));
  }
  return absl::OkStatus();
}

void TfDistributionManager::WorkerRun(Blob blob, Worker* worker) {
  auto result = worker->connection->RunTask(std::move(blob));
  if (verbose_ && !result.ok()) {
    LOG(WARNING) << "Session called failed with error: "
                 << result.status().ToString();
  }
  async_pending_answers_.Push(std::move(result));
}

void TfDistributionManager::WorkerMainLocalPool(Worker* worker) {
  while (true) {
    auto pending_blob_or = worker->async_pending_queries_.Pop();
    if (!pending_blob_or.has_value()) {
      break;
    }
    WorkerRun(std::move(pending_blob_or.value()), worker);
  }
}

void TfDistributionManager::WorkerMainGlobalPool(Worker* worker) {
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
  if (verbose_) {
    LOG(INFO) << "Sending blocking request with " << blob.size() << " bytes";
  }
  if (worker_idx < 0) {
    worker_idx = next_auto_worker_idx_.fetch_add(1) % workers_.size();
  }
  auto* worker = workers_[worker_idx].get();
  return worker->connection->RunTask(std::move(blob));
}

absl::Status TfDistributionManager::AsynchronousRequest(Blob blob,
                                                        int worker_idx) {
  if (verbose_) {
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
  if (verbose_) {
    LOG(INFO) << "Wait for next asynchronous result";
  }
  auto answer_or = async_pending_answers_.Pop();
  if (!answer_or.has_value()) {
    return absl::InvalidArgumentError("No more results available");
  }
  if (verbose_) {
    LOG(INFO) << "Receive asynchronous request with "
              << answer_or.value()->size() << " bytes";
  }
  return std::move(*answer_or.value());
}

int TfDistributionManager::NumWorkers() { return workers_.size(); }

absl::Status TfDistributionManager::Done(
    absl::optional<bool> kill_worker_manager) {
  if (verbose_) {
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

  JoinWorkers();

  // TODO: Run in parallel.
  for (auto& worker : workers_) {
    auto worker_shutdown =
        worker->connection->StopWorker(kill_worker_manager.value_or(false));
    if (!worker_shutdown.ok()) {
      // It is not a big deal if the worker crashes during shutdown.
      LOG(WARNING) << worker_shutdown.error_message();
    }
  }

  workers_.clear();

  if (verbose_) {
    LOG(INFO) << "Manager has been shutdown";
  }

  return absl::OkStatus();
}

void TfDistributionManager::JoinWorkers() {
  for (auto& worker : workers_) {
    worker->main_thread_local_pool->Join();
    worker->main_thread_global_pool->Join();
  }
}

absl::Status TfDistributionManager::Initialize(
    const proto::Config& config, const absl::string_view worker_name,
    Blob welcome_blob) {
  verbose_ = config.verbose();

  if (verbose_) {
    LOG(INFO) << "Initialize manager with " << welcome_blob.size()
              << " bytes welcome blob";
  }

  RETURN_IF_ERROR(
      InitializeWorkers(config, worker_name, std::move(welcome_blob)));
  return absl::OkStatus();
}

namespace internal {

tf::Status RemoteConnection::Create(
    const std::string& target, const Blob& welcome_blob,
    const absl::string_view worker_name, const int worker_idx,
    std::unique_ptr<RemoteConnection>* connection) {
  // Generate manager uid.  Used to distinguish between the different managers
  // controlling a same pool of workers.
  std::random_device rnd;
  auto rnd_value = std::uniform_int_distribution<uint64_t>(
      std::numeric_limits<uint64_t>::lowest(),
      std::numeric_limits<uint64_t>::max())(rnd);

  // Identifier of the tf resource containing the worker.
  std::string resource_uid =
      absl::Substitute("$0_$1_$2_$3", worker_name, worker_idx,
                       absl::GetCurrentTimeNanos(), rnd_value);

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
            .Attr("resource_uid", resource_uid);
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
            .Attr("resource_uid", resource_uid);
    conn->root->UpdateBuilder(&builder);
    conn->root->UpdateStatus(
        builder.Finalize(conn->root->graph(), &conn->stop_worker_node));
    if (!conn->root->ok()) return conn->root->status();

    conn->stop_worker_op = tf::Operation(conn->stop_worker_node);
  }

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

utils::StatusOr<Blob> RemoteConnection::RunTask(Blob&& input) {
  tf::ClientSession::FeedType feeds;
  feeds.emplace(tf::Output(*run_task_input), tf::Input::Initializer(input));
  std::vector<tf::Tensor> outputs;
  RETURN_IF_ERROR(
      utils::ToUtilStatus(session->Run(feeds, {run_task_output}, &outputs)));
  return outputs[0].flat<tf::tstring>()(0);
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
    workers.push_back(worker_item_it->GetString());
  }

  return workers;
}

}  // namespace internal

}  // namespace distribute
}  // namespace yggdrasil_decision_forests
