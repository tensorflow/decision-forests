#!/bin/bash
# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



# Simulate distributed training locally.
#

echo "Warning: This script does not stop workers. After running it, you need to"
echo "stop the workers manually (e.g. using kill -9)."

set -vex

PYTHON=python3

export TF_CONFIG='{
"cluster": {
  "worker": [
    "localhost:4300",
    "localhost:4301"],
  "ps": ["localhost:4310"],
  "chief": ["localhost:4311"]
}
}'

bazel build \
  //third_party/tensorflow_decision_forests/examples:distributed_training \
  //third_party/tensorflow_decision_forests/tensorflow/distribute:tensorflow_std_server_py

WORKER=bazel-bin/third_party/tensorflow_decision_forests/tensorflow/distribute/tensorflow_std_server_py
CHIEF=bazel-bin/third_party/tensorflow_decision_forests/examples/distributed_training

# Start the workers
${WORKER} --alsologtostderr --job_name=worker --task_index=0 &
${WORKER} --alsologtostderr --job_name=worker --task_index=1 &
${WORKER} --alsologtostderr --job_name=ps --task_index=0 &

# Start the chief
${CHIEF} --alsologtostderr
