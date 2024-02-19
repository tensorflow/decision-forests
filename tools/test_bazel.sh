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
#              For cross-compiling with Apple Silicon for Mac Intel, use 
#              mac-intel-crosscompile.
#              Tests will not work when cross-compiling (obviously).
# FULL_COMPILATION: If 1, compile all parts of TF-DF. This may take a long time.
#
# Usage example
#
#   RUN_TESTS=1 PY_VERSION=3.9 TF_VERSION=2.13.0 ./tools/test_bazel.sh

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

# Install Pip dependencies
${PYTHON} -m ensurepip --upgrade || true
${PYTHON} -m pip install pip setuptools --upgrade
${PYTHON} -m pip install numpy pandas scikit-learn

# Install Tensorflow at the chosen version.
if [ ${TF_VERSION} == "nightly" ]; then
  ${PYTHON} -m pip install tf-nightly tf-keras-nightly --force-reinstall
  TF_MINOR="nightly"
else
  ${PYTHON} -m pip install tensorflow==${TF_VERSION} --force-reinstall
  TF_MINOR=$(echo $TF_VERSION | grep -oP '[0-9]+\.[0-9]+')
  ${PYTHON} -m pip install tf-keras==${TF_MINOR}  --force-reinstall
fi
ext=""

pip list

if is_macos; then
  ext='""'
  # Tensorflow requires the use of GNU realpath instead of MacOS realpath.
  # See https://github.com/tensorflow/tensorflow/issues/60088#issuecomment-1499766349
  # If missing, install coreutils via homebrew: `brew install coreutils`
  export PATH="/opt/homebrew/opt/coreutils/libexec/gnubin:$PATH"
fi

# For Tensorflow versions > 2.15, apply compatibility patches.

if [[ ${TF_MINOR} != "2.15" ]]; then
  sed -i $ext "s/tensorflow:tf.patch/tensorflow:tf-216.patch/" WORKSPACE
  sed -i $ext "s/# patch_args = \[\"-p1\"\],/patch_args = \[\"-p1\"\],/" third_party/yggdrasil_decision_forests/workspace.bzl
  sed -i $ext "s/# patches = \[\"\/\/third_party\/yggdrasil_decision_forests:ydf.patch\"\],/patches = \[\"\/\/third_party\/yggdrasil_decision_forests:ydf.patch\"\],/" third_party/yggdrasil_decision_forests/workspace.bzl
fi

# Get the commit SHA
short_commit_sha=$(${PYTHON} -c 'import tensorflow as tf; print(tf.__git_version__)' | tail -1)
if is_macos; then
  short_commit_sha=$(echo $short_commit_sha | perl -nle 'print $& while m{(?<=-g)[0-9a-f]*$}g')
else
  short_commit_sha=$(echo $short_commit_sha | grep -oP '(?<=-g)[0-9a-f]*$')
fi
echo "Found tensorflow commit sha: $short_commit_sha"
commit_slug=$(curl -s "https://api.github.com/repos/tensorflow/tensorflow/commits/$short_commit_sha" | grep "sha" | head -n 1 | cut -d '"' -f 4)
# Update TF dependency to the chosen version
sed -E -i $ext "s/strip_prefix = \"tensorflow-2\.[0-9]+\.[0-9]+(-rc[0-9]+)?\",/strip_prefix = \"tensorflow-${commit_slug}\",/" WORKSPACE
sed -E -i $ext "s|\"https://github.com/tensorflow/tensorflow/archive/v.+\.zip\"|\"https://github.com/tensorflow/tensorflow/archive/${commit_slug}.zip\"|" WORKSPACE
prev_shasum=$(grep -A 1 -e "strip_prefix.*tensorflow-" WORKSPACE | tail -1 | awk -F '"' '{print $2}')
sed -i $ext "s/sha256 = \"${prev_shasum}\",//" WORKSPACE

# Get build configuration for chosen version.
TENSORFLOW_BAZELRC="tensorflow_bazelrc"
curl https://raw.githubusercontent.com/tensorflow/tensorflow/${commit_slug}/.bazelrc -o ${TENSORFLOW_BAZELRC}

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
  FLAGS="${FLAGS} --config=linux"
fi

if [ ${TF_VERSION} == "mac-intel-crosscompile" ]; then
  TFDF_TMPDIR="${TMPDIR}tf_dep"
  rm -rf ${TFDF_TMPDIR}
  mkdir -p ${TFDF_TMPDIR}
  # Download the Intel CPU Tensorflow package
  pip download --no-deps --platform=macosx_10_15_x86_64 --dest=$TFDF_TMPDIR tensorflow
  unzip -q $TFDF_TMPDIR/tensorflow* -d $TFDF_TMPDIR

  # Find the path to the pre-compiled version of TensorFlow installed in the
  # "tensorflow" pip package.
  SHARED_LIBRARY_DIR=$(readlink -f $TFDF_TMPDIR/tensorflow)
  SHARED_LIBRARY_NAME="libtensorflow_cc.2.dylib"
  HEADER_DIR=$(readlink -f $TFDF_TMPDIR/tensorflow/include)
else
  # Find the path to the pre-compiled version of TensorFlow installed in the
  # "tensorflow" pip package.
  TF_CFLAGS="$(${PYTHON} -c 'import tensorflow as tf; print(tf.sysconfig.get_compile_flags()[0])')"
  TF_LFLAGS="$(${PYTHON} -c 'import tensorflow as tf; print(tf.sysconfig.get_link_flags()[0])')"

  HEADER_DIR=${TF_CFLAGS:2}
  if is_macos; then
    SHARED_LIBRARY_DIR=${TF_LFLAGS:2}
    SHARED_LIBRARY_NAME="libtensorflow_framework.2.dylib"
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
fi

FLAGS="${FLAGS} --action_env TF_HEADER_DIR=${HEADER_DIR}"
FLAGS="${FLAGS} --action_env TF_SHARED_LIBRARY_DIR=${SHARED_LIBRARY_DIR}"
FLAGS="${FLAGS} --action_env TF_SHARED_LIBRARY_NAME=${SHARED_LIBRARY_NAME}"

# Bazel
BAZEL=bazel

STARTUP_FLAGS="${STARTUP_FLAGS} --bazelrc=${TENSORFLOW_BAZELRC}"

# Distributed compilation using Remote Build Execution (RBE)
#
# FLAGS="$FLAGS --config=rbe_cpu_linux --config=tensorflow_testing_rbe_linux --config=rbe_linux_py3"

if [ ${TF_VERSION} == "mac-intel-crosscompile" ]; then
  # Using darwin_x86_64 fails here, tensorflow expects "darwin".
  FLAGS="${FLAGS} --cpu=darwin --apple_platform_type=macos"
fi

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
