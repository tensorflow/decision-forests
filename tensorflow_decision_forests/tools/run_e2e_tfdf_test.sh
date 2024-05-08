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



# Converts a non-submitted CL to a standalone Bazel project in a local
# directory, compile the project and run the tests.
#
# Usage example:
#   third_party/tensorflow_decision_forests/tools/run_e2e_tfdf_test.sh

set -vex

LOCAL_DIR="/usr/local/google/home/${USER}/git/decision-forests"

CL=$(hg exportedcl)
echo "Current CL: ${CL}"
echo "Make sure the CL is synced!"

function export_project() {
  COPYBARA="/google/bin/releases/copybara/public/copybara/copybara"

  # Test the copy bara configuration.
  bazel test third_party/tensorflow_decision_forests:copybara_test

  echo "Export a Bazel project locally"
  echo "=============================="

  rm -fr ${LOCAL_DIR}
  ${COPYBARA} third_party/tensorflow_decision_forests/copy.bara.sky presubmit_piper_to_gerrit ${CL} \
    --dry-run --init-history --squash --force \
    --git-destination-path ${LOCAL_DIR} --ignore-noop

  /google/bin/releases/opensource/thirdparty/cross/cross ${LOCAL_DIR}
}

echo "Test the project"
echo "================"

run_all() {
  cd ${LOCAL_DIR}

  # Start the Docker
  sudo ./tools/start_compile_docker.sh /bin/bash
  
  # In the docker, you can now trigger the builder with the following line in
  # the docker:
  # RUN_TESTS=1 PY_VERSION=3.9 TF_VERSION=2.16.1 ./tools/test_bazel.sh

  # Alternatively, you can trigger the build directly with:
  # sudo ./tools/start_compile_docker.sh "RUN_TESTS=1 PY_VERSION=3.8 TF_VERSION=2.10.0 ./tools/test_bazel.sh && chmod -R a+xrw . && /bin/bash"
}

export_project
run_all
