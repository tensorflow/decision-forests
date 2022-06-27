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



set -e  # fail and exit on any command erroring
set -x  # print evaluated commands

PY_VERSION=${1}
TF_BRANCH="latest"

function is_nightly() {
  [[ "$IS_NIGHTLY" == "nightly" ]]
}

# cd into the release branch in kokoro
cd "${KOKORO_ARTIFACTS_DIR}"/git/tensorflow_decision_forests/

perl -0777 -i.original -pe 's/    http_archive\(\n        name = "ydf",\n        urls = \["https:\/\/github.com\/google\/yggdrasil-decision-forests\/archive\/refs\/heads\/main.zip"\],\n        strip_prefix = "yggdrasil-decision-forests-main",\n    \)/    native.local_repository\(\n        name = "ydf",\n        path = "..\/yggdrasil_decision_forests",\n    \)/igs' third_party/yggdrasil_decision_forests/workspace.bzl

# Pull docker image specific to python and tensorflow version
docker pull tensorflow/build:${TF_BRANCH}-python${PY_VERSION}

# Run docker container. Container name => tfdf_container.
docker run --privileged --name tfdf_container -w /working_dir/tensorflow_decision_forests \
  -itd --rm \
  -v "$KOKORO_GFILE_DIR:/kokoro_gfile_dir" \
  -v "$KOKORO_ARTIFACTS_DIR/git/:/working_dir" \
  tensorflow/build:${TF_BRANCH}-python${PY_VERSION} \
  bash

docker exec tfdf_container /usr/bin/python3 -m pip install --upgrade pip
docker exec tfdf_container pip install tensorflow numpy pandas scikit-learn --upgrade

# List all installed packages and versions present inside container
echo -e "pip installed packages: \n"
docker exec tfdf_container pip list

docker exec tfdf_container tools/test_bazel.sh

docker stop tfdf_container
