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

#include "tensorflow_decision_forests/tensorflow/distribute/tf_distribution.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow_decision_forests/tensorflow/distribute/tf_distribution.pb.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.h"
#include "yggdrasil_decision_forests/utils/distribute/test_utils.h"
#include "yggdrasil_decision_forests/utils/distribute/toy_worker.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace distribute {
namespace {

// Create a TF DIst manager and its workers.
ManagerAndWorkers CreateTfDistManager(int parallel_execution_per_worker = 1) {
  ManagerAndWorkers manager_and_workers;
  // Manager configuration.
  proto::Config config;
  config.set_implementation_key("TF_DIST");
  config.set_verbosity(2);
  config.set_working_directory(
      file::JoinPath(test::TmpDirectory(), "work_dir"));
  auto* addresses =
      config.MutableExtension(proto::tf_distribution)->mutable_addresses();
  const int num_workers = 5;

  // Create the TF Server config in JSON format.
  std::string cluster_json = "\"cluster\": { \"worker\": [";
  for (int worker_idx = 0; worker_idx < num_workers; worker_idx++) {
    // Create address.
    const int port = tensorflow::testing::PickUnusedPortOrDie();
    CHECK_GT(port, 0);
    const std::string address = absl::StrCat("localhost:", port);
    addresses->add_addresses(absl::StrCat("grpc://", address));

    if (worker_idx > 0) {
      absl::StrAppend(&cluster_json, ",");
    }
    absl::SubstituteAndAppend(&cluster_json, "\"$0\"", address);
  }
  absl::StrAppend(&cluster_json, "]}");

  // Start the workers.
  const auto binary =
      file::JoinPath(test::DataRootDirectory(),
                     "tensorflow_decision_forests/tensorflow/"
                     "distribute/tf_distribution_py_worker");

  for (int worker_idx = 0; worker_idx < num_workers; worker_idx++) {
    std::string env_TF_CONFIG = absl::Substitute(
        "{$0, \"task\": {\"type\": \"worker\", \"index\": $1}}", cluster_json,
        worker_idx);
    LOG(INFO) << "Env config:" << env_TF_CONFIG;

    setenv("TF_CONFIG", env_TF_CONFIG.c_str(), 1);
    const auto command = absl::Substitute("$0 --alsologtostderr", binary);

    // Create worker thread.
    manager_and_workers.worker_threads.push_back(
        absl::make_unique<utils::concurrency::Thread>(
            [command]() { CHECK_EQ(system(command.c_str()), 0); }));

    // Make sure we don't override the env variable.
    // TODO(gbm): Find a better way.
    absl::SleepFor(absl::Seconds(0.5));
  }

  // Wait for the servers to start.
  absl::SleepFor(absl::Seconds(2));

  // Start manager.
  manager_and_workers.manager = CreateManager(config, kToyWorkerKey, "hello",
                                              parallel_execution_per_worker)
                                    .value();
  return manager_and_workers;
}

TEST(TFDist, ParseEnv) {
  std::string content = R"(
{
   "cluster":{
      "chief":[
         "chief/0"
      ],
      "ps":[
         "ps/0"
      ],
      "worker":[
         "worker/0",
         "worker/1",
         "worker/2"
      ]
   },
   "environment":"google",
   "rpc_layer":"grpc+loas",
   "task":{
      "index":"%task%",
      "type":"worker"
   }
}
)";
  EXPECT_EQ(
      internal::JsonConfigToWorkers(content).value(),
      std::vector<std::string>({"grpc+loas://worker/0", "grpc+loas://worker/1",
                                "grpc+loas://worker/2"}));
}

TEST(TFDist, BlockingRequest) {
  auto all = CreateTfDistManager();
  TestBlockingRequest(all.manager.get());
  all.Join();
}

TEST(TFDist, WorkerError) {
  auto all = CreateTfDistManager();
  TestBlockingRequest(all.manager.get());
  all.Join();
}

TEST(TFDist, AsynchronousRequest) {
  auto all = CreateTfDistManager();
  TestAsynchronousRequest(all.manager.get());
  all.Join();
}

TEST(TFDist, BlockingRequestWithSpecificWorker) {
  auto all = CreateTfDistManager();
  TestBlockingRequestWithSpecificWorker(all.manager.get());
  all.Join();
}

TEST(TFDist, AsynchronousRequestWithSpecificWorker) {
  auto all = CreateTfDistManager();
  TestAsynchronousRequestWithSpecificWorker(all.manager.get());
  all.Join();
}

// Enable the following test when the TF-Distribution supports worker-to-worker
// communication.
TEST(TFDist, AsynchronousIntraWorkerCommunication) {
  auto all = CreateTfDistManager();
  TestAsynchronousIntraWorkerCommunication(all.manager.get());
  all.Join();
}

TEST(TFDist, AsynchronousParallelWorkerExecution) {
  auto all = CreateTfDistManager(5);
  TestAsynchronousParallelWorkerExecution(all.manager.get());
  all.Join();
}

}  // namespace
}  // namespace distribute
}  // namespace yggdrasil_decision_forests
