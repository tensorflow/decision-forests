# Installation

<!-- docs_infra:strip_begin -->

## Table of Contents

<!--ts-->

*   [Installation](#installation)
    *   [Table of Contents](#table-of-contents)
    *   [Installation with pip](#installation-with-pip)
    *   [Build from source](#build-from-source)
    *   [Troubleshooting](#troubleshooting)

<!-- Added by: gbm, at: Mon 10 May 2021 03:50:43 PM CEST -->

<!--te-->

<!-- docs_infra:strip_end -->

## Installation with pip

To install TensorFlow Decision Forests, run:

```shell
pip3 install tensorflow_decision_forests --upgrade
```

Then, check the installation with:

```shell
python3 -c "import tensorflow_decision_forests as tfdf; print('Found TF-DF v' + tfdf.__version__)"
```

**Note:** Cuda warnings are not an issue.

## Build from source

**Requirements**

-   The [requirements](https://www.tensorflow.org/install/source) of TensorFlow.
-   The [requirements](https://github.com/google/yggdrasil-decision-forests) of
    Yggdrasil Decision Forests.

**Requirements (Details)**

-   MSVC>=2019 (Windows) or GCC>=7.3 (Linux).
-   Bazel 3.7.2
-   Python 3.x
-   Git
-   Python's numpy
-   msys64 (Windows only)

Download and compile TensorFlow Decision Forests as follow:

```shell
# Download the source code.
git clone https://github.com/tensorflow/decision-forests.git
cd decision-forests

# Download TensorFlow's BazelRC.
# See "tensorflow_bazelrc" for the compilation options.
# Note: If you compile with an older version of TensorFlow, download the
# corresponding bazelrc.
TENSORFLOW_BAZELRC="tensorflow_bazelrc"
wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/.bazelrc -O ${TENSORFLOW_BAZELRC}

# Compile TensorFlow Decision Forests.
bazel-3.7.2 --bazelrc=${TENSORFLOW_BAZELRC} build //tensorflow_decision_forests/...:all --config=linux

# Create a pip package.
./tools/build_pip_package.sh simple

# Install the pip package
pip3 install dist/*.whl
```

**Note:** Compiling TensorFlow Decision Forests compiles a large part of
TensorFlow. This operation will take multiple hours without distributed
building.

## Troubleshooting

**Note:** Check also Yggdrasil's
[Troubleshooting](https://github.com/google/yggdrasil-decision-forests/manual/installation.md#Troubleshooting)
page.
