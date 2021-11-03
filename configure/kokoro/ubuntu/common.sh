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



set -e
set -x

# Error if we somehow forget to set the path to bazel_wrapper.py
set -u
BAZEL_WRAPPER_PATH=$1
set +u

source tensorflow/tools/ci_build/release/common.sh
install_bazelisk
which bazel

# Run bazel test command.
"${BAZEL_WRAPPER_PATH}" \
  test \
  --config=rbe_cpu_linux \
  --config=rbe_linux_py3 \
  --define tf_ps_distribution_strategy=0 \
  --python_path="/usr/bin/python3.9" \
  --config=tensorflow_testing_rbe_linux \
  -- \
  //tensorflow_decision_forests/...:all

# Copy log to output to be available to GitHub
ls -la "$(bazel info output_base)/java.log"
cp "$(bazel info output_base)/java.log" "${KOKORO_ARTIFACTS_DIR}/"
