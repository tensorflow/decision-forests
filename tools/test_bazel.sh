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



# Build and test TF-DF.

set -vex

# Version of Python
# Needs to be >=python3.7
PYTHON=python3.8

# Install Pip dependencies
${PYTHON} -m ensurepip --upgrade || true
${PYTHON} -m pip install pip --upgrade
${PYTHON} -m pip install tensorflow numpy pandas --upgrade

# Force a compiler
# export CC=gcc-8
# export CXX=gcc-8

# Running flags.
FLAGS=
STARTUP_FLAGS=

# Detect the target host
PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"

function is_windows() {
  # On windows, the shell script is actually running in msys
  [[ "${PLATFORM}" =~ msys_nt*|mingw*|cygwin*|uwin* ]]
}

function is_macos() {
  [[ "${PLATFORM}" == "darwin" ]]
}

if is_macos; then
  FLAGS="--config=macos --config=release_cpu_macos"
elif is_windows; then
  FLAGS="--config=windows --config=release_cpu_windows"
else
  FLAGS="--config=linux --config=release_cpu_linux"
fi

# Find the path to the pre-compiled version of TensorFlow installed in the
# "tensorflow" pip package.
TF_CFLAGS=( $(${PYTHON} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS="$(${PYTHON} -c 'import tensorflow as tf; print(tf.sysconfig.get_link_flags()[0])')"

HEADER_DIR=${TF_CFLAGS:2}
if is_macos; then
  SHARED_LIBRARY_NAME="libtensorflow_framework.dylib"
  SHARED_LIBRARY_DIR=${TF_LFLAGS:2}
elif is_windows; then
 # Use pywrap_tensorflow's import library on Windows. It is in the same dir as the dll/pyd.
  SHARED_LIBRARY_NAME="_pywrap_tensorflow_internal.lib"
  SHARED_LIBRARY_DIR=${TF_CFLAGS:2:-7}"python"

  SHARED_LIBRARY_NAME=${SHARED_LIBRARY_NAME//\\//}
  SHARED_LIBRARY_DIR=${SHARED_LIBRARY_DIR//\\//}
  HEADER_DIR=${HEADER_DIR//\\//}
else
  SHARED_LIBRARY_DIR=${TF_LFLAGS:2}
  SHARED_LIBRARY_NAME="libtensorflow_framework.so.2"
fi

FLAGS="${FLAGS} --action_env TF_HEADER_DIR=${HEADER_DIR}"
FLAGS="${FLAGS} --action_env TF_SHARED_LIBRARY_DIR=${SHARED_LIBRARY_DIR}"
FLAGS="${FLAGS} --action_env TF_SHARED_LIBRARY_NAME=${SHARED_LIBRARY_NAME}"

# Bazel
#
# Note: TensorFlow is not (Mar2021) compatible with Bazel4.
BAZEL=bazel

# TensorFlow building configuration
#
# Note: Copy the building configuration of TF.
TENSORFLOW_BAZELRC="tensorflow_bazelrc"
wget https://raw.githubusercontent.com/tensorflow/tensorflow/v2.7.0/.bazelrc -O ${TENSORFLOW_BAZELRC}
STARTUP_FLAGS="${STARTUP_FLAGS} --bazelrc=${TENSORFLOW_BAZELRC}"

# Distributed compilation using Remote Build Execution (RBE)
#
# copybara:strip_begin
# First follow the instruction: go/tf-rbe-guide
# copybara:strip_end
# FLAGS="$FLAGS --config=rbe_cpu_linux --config=tensorflow_testing_rbe_linux --config=rbe_linux_py3"

# Minimal rules to create and test the Pip Package.
#
# Only require a small amount of TF to be compiled.
BUILD_RULES="//tensorflow_decision_forests/component/...:all //tensorflow_decision_forests/keras //tensorflow_decision_forests/keras:grpc_worker_main"
TEST_RULES="//tensorflow_decision_forests/component/...:all //tensorflow_decision_forests/keras:keras_test"

# All the build rules.
#
# BUILD_RULES="//tensorflow_decision_forests/...:all"
# TEST_RULES="//tensorflow_decision_forests/...:all"

# Disable distributed training with TF Parameter Server
#
# Note: Currently, distributed training with parameter server is only supported
# in the monolithic build. Distributed training is available with the Yggdrasil
# Distribution through.
FLAGS="${FLAGS} --define tf_ps_distribution_strategy=0"
# TEST_RULES="${TEST_RULES} //tensorflow_decision_forests/keras:keras_distributed_test"

# Build library
time ${BAZEL} ${STARTUP_FLAGS} build ${BUILD_RULES} ${FLAGS}

# Unit test library
time ${BAZEL} ${STARTUP_FLAGS} test ${TEST_RULES} ${FLAGS}

# Example of dependency check.
# ${BAZEL} --bazelrc=${TENSORFLOW_BAZELRC} cquery "somepath(//tensorflow_decision_forests/tensorflow/ops/inference:api_py,@org_tensorflow//tensorflow/c:kernels.cc)" ${FLAGS}
