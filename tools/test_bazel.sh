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


# Compile and runs the unit tests.

set -x
set -e

# Note: TensorFlow is not (Mar2021) compatible with Bazel4.
BAZEL=bazel-3.7.2

# Distributed compilation using RBE i.e. a remove server (fast).
# Set the following variable to tensorflow's bashrc. You might have to download
# this file from the github (https://github.com/tensorflow/tensorflow).
# TENSORFLOW_BAZELRC="${HOME}/git/tf_bazelrc"

# Alternatively, download bazelrc:
TENSORFLOW_BAZELRC="tensorflow_bazelrc"
wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/.bazelrc -O ${TENSORFLOW_BAZELRC}

# copybara:strip_begin
# First follow the instruction: go/tf-rbe-guide
# copybara:strip_end

FLAGS="--config=linux --config=rbe_linux_py3 --config=tensorflow_testing_rbe_linux --config=rbe_cpu_linux"

# Uncomment the following line to generate a sharable pip package.
# You will also need to install the dockers described in:
# https://github.com/tensorflow/custom-op
# FLAGS="${FLAGS} --crosstool_top=//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010-nvcc-cuda10.1:toolchain"

${BAZEL} --bazelrc=${TENSORFLOW_BAZELRC} build \
  //tensorflow_decision_forests/...:all \
  ${FLAGS}

${BAZEL} --bazelrc=${TENSORFLOW_BAZELRC} test \
  //tensorflow_decision_forests/...:all \
  ${FLAGS}
