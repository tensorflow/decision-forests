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

function print_kokoro_vars () {
  # Setup Bazel from x20, should read local bazel in production
  echo KOKORO_BAZEL_AUTH_CREDENTIAL: "${KOKORO_BAZEL_AUTH_CREDENTIAL}"
  echo KOKORO_BAZEL_TLS_CREDENTIAL: "${KOKORO_BAZEL_TLS_CREDENTIAL}"
  echo KOKORO_BES_BACKEND_ADDRESS: "${KOKORO_BES_BACKEND_ADDRESS}"
  echo KOKORO_BES_PROJECT_ID: "${KOKORO_BES_PROJECT_ID}"
  echo KOKORO_FOUNDRY_BACKEND_ADDRESS: "${KOKORO_FOUNDRY_BACKEND_ADDRESS}"
}

function setup_internal_tools () {
  # Source the internal common scripts. This file must not change between
  # releases, except in exceptional cases.
  source "${KOKORO_PIPER_DIR}/google3/learning/brain/testing/kokoro/common_google.sh"

  # Make bazel_wrapper.py executable. Another file which must not change between
  # branches, except in exceptional cases.
  chmod +x "${KOKORO_PIPER_DIR}/google3/learning/brain/testing/kokoro/bazel_wrapper.py"

  # Setup .bazelrc file
  setup_workspace_multi_scm
}

# Private part
print_kokoro_vars
setup_internal_tools

cd "${KOKORO_ARTIFACTS_DIR}/git/tensorflow_decision_forests"
./configure/kokoro/ubuntu/common.sh "${KOKORO_PIPER_DIR}/google3/learning/brain/testing/kokoro/bazel_wrapper.py"
