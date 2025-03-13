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
# Options
#  RUN_TESTS: Run the unit tests e.g. 0 or 1.
#  PY_VERSION: Version of Python to be used, must be at least 3.9
#  STARTUP_FLAGS: Any flags given to bazel on startup
#  TF_VERSION: Tensorflow version to use or "nightly".
#  MAC_INTEL_CROSSCOMPILE: Cross-compile for Intel Macs
#  FULL_COMPILATION: If 1, compile all parts of TF-DF. This may take a long time.
#
# Usage example
#
#   RUN_TESTS=1 PY_VERSION=3.9 TF_VERSION=2.19.0 ./tools/test_bazel.sh

set -vex

# Version of Python
# Needs to be >=python3.9
PYTHON=python${PY_VERSION}

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
  # 1. Check if the current shell is Bash
  if [[ $SHELL != "/bin/bash" ]]; then
      echo "Error: This script requires Bash. Please run it in a Bash shell."
      exit 1  # Exit with an error code
  fi
  if ! command -v gsort &> /dev/null; then  
      echo "Error: GNU coreutils is not installed. Please install it with 'brew install coreutils'"
      exit 1
  fi
  if ! command -v ggrep &> /dev/null; then
      echo "Error: GNU grep is not installed. Please install it with 'brew install grep'"
      exit 1
  fi
  if ! command -v gsed &> /dev/null; then
      echo "Error: GNU sed is not installed. Please install it with 'brew install gnu-sed'"
      exit 1
  fi
  # Tensorflow requires the use of GNU realpath instead of MacOS realpath.
  # See https://github.com/tensorflow/tensorflow/issues/60088#issuecomment-1499766349
  export PATH="/opt/homebrew/opt/coreutils/libexec/gnubin:$PATH"
  export PATH="/opt/homebrew/opt/grep/libexec/gnubin:$PATH"
  export PATH="/opt/homebrew/opt/gnu-sed/libexec/gnubin:$PATH"
fi

# Install Tensorflow at the chosen version.
if [ ${TF_VERSION} == "nightly" ]; then
  ${PYTHON} -m pip install -q tf-nightly --force-reinstall
  sed -i "s/^tensorflow==.*/tf-nightly/" configure/requirements.in
  sed -i "s/^tf-keras.*/tf-keras-nightly/" configure/requirements.in
  export IS_NIGHTLY=1
else
  ${PYTHON} -m pip install -q tensorflow==${TF_VERSION} --force-reinstall
  sed -i "s/^tensorflow==.*/tensorflow==$TF_VERSION/" configure/requirements.in
fi

# Get the commit SHA
short_commit_sha=$(${PYTHON} -c 'import tensorflow as tf; print(tf.__git_version__)' | tail -1)
short_commit_sha=$(echo $short_commit_sha | grep -oP '(?<=-g)[0-9a-f]*$')
echo "Found tensorflow commit sha: $short_commit_sha"
commit_slug=$(curl -s "https://api.github.com/repos/tensorflow/tensorflow/commits/$short_commit_sha" | grep "sha" | head -n 1 | cut -d '"' -f 4)
# Update TF dependency to the chosen version
sed -E -i "s/strip_prefix = \"tensorflow-2\.[0-9]+(\.[0-9]+)*(-rc[0-9]+)?\",/strip_prefix = \"tensorflow-${commit_slug}\",/" WORKSPACE
sed -E -i "s|\"https://github.com/tensorflow/tensorflow/archive/v.+\.tar.gz\"|\"https://github.com/tensorflow/tensorflow/archive/${commit_slug}.tar.gz\"|" WORKSPACE
prev_shasum=$(grep -B 1 -e "strip_prefix.*tensorflow-" WORKSPACE | head -1 | awk -F '"' '{print $2}')
sed -i "s/sha256 = \"${prev_shasum}\",//" WORKSPACE

# Get build configuration for chosen version.
TENSORFLOW_BAZELRC="tensorflow_bazelrc"
curl https://raw.githubusercontent.com/tensorflow/tensorflow/${commit_slug}/.bazelrc -o ${TENSORFLOW_BAZELRC}

echo "TF_VERSION=$TF_VERSION"
REQUIREMENTS_EXTRA_FLAGS="--upgrade"
if [[ "$TF_VERSION" == *"rc"* ]]; then
  REQUIREMENTS_EXTRA_FLAGS="$REQUIREMENTS_EXTRA_FLAGS --pre"
fi

bazel run //configure:requirements.update -- $REQUIREMENTS_EXTRA_FLAGS

# Bazel common flags. Startup flags are already given through STARTUP_FLAGS
FLAGS=

if is_macos; then
  FLAGS="${FLAGS} --config=macos"
  if [[ $(uname -m) == 'arm64' ]]; then
    FLAGS="${FLAGS} --config=macos_arm64"
  else
    FLAGS="${FLAGS} --config=macos_intel"
  fi
elif is_windows; then
  FLAGS="${FLAGS} --config=windows"
else
  FLAGS="${FLAGS} --config=linux --config=release_cpu_linux"
fi


# Bazel
BAZEL=bazel

STARTUP_FLAGS="${STARTUP_FLAGS} --bazelrc=${TENSORFLOW_BAZELRC}"

# Distributed compilation using Remote Build Execution (RBE)
#
# FLAGS="$FLAGS --config=rbe_cpu_linux --config=tensorflow_testing_rbe_linux --config=rbe_linux_py3"

if [ "${FULL_COMPILATION}" == 1 ]; then
  BUILD_RULES="//tensorflow_decision_forests/...:all"
  TEST_RULES="//tensorflow_decision_forests/...:all"
else
  BUILD_RULES="//tensorflow_decision_forests/component/...:all //tensorflow_decision_forests/contrib/...:all //tensorflow_decision_forests/keras //tensorflow_decision_forests/keras:grpc_worker_main"
  TEST_RULES="//tensorflow_decision_forests/component/...:all //tensorflow_decision_forests/contrib/...:all //tensorflow_decision_forests/keras/...:all"
fi

# Build library
time ${BAZEL} ${STARTUP_FLAGS} build ${BUILD_RULES} ${FLAGS}

# Unit test library
if [ "${RUN_TESTS}" == 1 ]; then
  time ${BAZEL} ${STARTUP_FLAGS} test ${TEST_RULES} ${FLAGS} --flaky_test_attempts=1 --test_size_filters=small,medium,large
fi

# Example of dependency check.
# ${BAZEL} --bazelrc=${TENSORFLOW_BAZELRC} cquery "somepath(//tensorflow_decision_forests/tensorflow/ops/inference:api_py,@org_tensorflow//tensorflow/c:kernels.cc)" ${FLAGS}
