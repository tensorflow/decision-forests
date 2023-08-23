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



# Start a docker able to compile TF-DF.
#
# Usage example
#
#  # Create the pip packages for publication
#
#  # Download and start the docker (this script)
#  ./tools/start_compile_docker.sh
#
#  # Compile and test TF-DF.
#  RUN_TESTS=1 PY_VERSION=3.8 TF_VERSION=2.12.0 ./tools/test_bazel.sh
#
#  # Create a Pip package for a specific version of python.
#  ./tools/build_pip_package.sh python3.8
#
#  # Install the other versions of python (the docker only has py3.8).
#  sudo apt-get update
#  sudo apt-get install python3.7 python3.9 python3-pip python3.10
#
#  # Create the Pip package for the other version of python
#  ./tools/build_pip_package.sh python3.7
#  ./tools/build_pip_package.sh python3.9
#  ./tools/build_pip_package.sh python3.10
#
#  # Make the result of the docker world accessible (in case the docker is run
#  # in root).
#  chmod -R a+xrw .
#
#  # Exit the docker
#  exit
#
#  # Publish the pip packages
#  ./tools/submit_pip_package.sh
#
# Alternative ending
#
#  # Create a Pip package for all the compatible version of pythons using pyenv.
#  ./tools/build_pip_package.sh ALL_VERSIONS
#
#  # Create a Pip package for all the compatible version of python using the
#  # previous build_pip_package call results (i.e. the "tmp_package" directory)
#  ./tools/build_pip_package.sh ALL_VERSIONS_ALREADY_ASSEMBLED
#
# https://hub.docker.com/r/tensorflow/build/tags?page=1
DOCKER=tensorflow/build:2.14-python3.8

# Current directory
# Useful if Yggdrasil Decision Forests is available locally in a neighbor
# directory.
TFDF_DIRNAME=${PWD##*/}

# Download docker
docker pull ${DOCKER}

# Start docker
docker run -it -v ${PWD}/..:/working_dir -w /working_dir/${TFDF_DIRNAME} ${DOCKER} $@
