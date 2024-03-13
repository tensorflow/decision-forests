# Installation

<!-- docs_infra:strip_begin -->

## Table of Contents

<!--ts-->

*   [Installation](#installation)
    *   [Table of Contents](#table-of-contents)
    *   [Installation with Pip](#installation-with-pip)
    *   [Build from source](#build-from-source)
        *   [Technical details](#technical-details)
        *   [Linux](#linux)
            *   [Docker build](#docker-build)
            *   [Manual build](#manual-build)
        *   [MacOS](#macos)
            *   [Setup](#setup)
            *   [Arm64 CPU](#arm64-cpu)
            *   [Cross-compiling for Intel CPUs](#cross-compiling-for-intel-cpus)
        *   [Windows](#windows)

<!--te-->

<!-- docs_infra:strip_end -->

## Installation with Pip

Install TensorFlow Decision Forests by running:

```shell
# Install TensorFlow Decision Forests.
pip3 install tensorflow_decision_forests --upgrade
```

Then, check the installation with by running:

```shell
# Check the version of TensorFlow Decision Forests.
python3 -c "import tensorflow_decision_forests as tfdf; print('Found TF-DF v' + tfdf.__version__)"
```

**Note:** Cuda warnings are not an issue.

## Build from source

### Technical details

TensorFlow Decision Forests (TF-DF) implements custom ops for TensorFlow and
therefore depends on TensorFlow's ABI. Since the ABI can change between
versions, any TF-DF version is only compatible with one specific TensorFlow
version.

To avoid compiling and shipping all of TensorFlow with TF-DF, TF-DF
links against libtensorflow shared library that is distributed with TensorFlow's
Pip package. Only a small part of Tensorflow is compiled and compilation only
takes ~10 minutes on a strong workstation (instead of multiple hours when
compiling all of TensorFlow). To ensure this works, the version of TensorFlow
that is actually compiled and the libtensorflow shared library must match
exactly.

The `tools/test_bazel.sh` script configures the TF-DF build to ensure the
versions of the packages used match. For details on this process, see the source
code of this script. Since TensorFlow compilation changes often, it only
supports building with the most recent TensorFlow versions and nightly.

**Note**: When distributing builds, you may set the `__git_version__` string in
`tensorflow_decision_forests/__init__.py` to identify the commit you built from.

### Linux

#### Docker build

The easiest way to build TF-DF on Linux is by using TensorFlow's build
[Build docker](https://github.com/tensorflow/build). Just run the following
steps to build:

```shell
./tools/start_compile_docker.sh # Start the docker, might require root
export RUN_TESTS=1              # Whether to run tests after build
export PY_VERSION=3.9           # Python version to use for build
# TensorFlow version to compile against. This must match exactly the version
# of TensorFlow used at runtime, otherwise TF-DF may crash unexpectedly.
export TF_VERSION=2.16.1        # Set to "nightly" for building with tf-nightly
./tools/test_bazel.sh
```

This places the compiled C++ code in the `bazel-bin` directory. Note that this
is a symbolic link that is not exposed outside the container (i.e. the build is
gone after leaving the container).

For building the wheels, run
```shell
tools/build_pip_package.sh ALL_VERSIONS INSTALL_PYENV
```

This will install [Pyenv](https://github.com/pyenv/pyenv) and
[Pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) inside the docker
and use it to install Python in all supported versions for building. The wheels
are placed in the `dist/` subdirectory.

#### Manual build

Building TF-DF without the docker might be harder, and the team is probably not
able to help with this.

**Requirements**

-   Bazel >= 6.3.0
-   Python >= 3
-   Git
-   Pyenv, Pyenv-virtualenv (only if packaging for many Python versions)

**Building**

Download TensorFlow Decision Forests as follows:

```shell
# Download the source code of TF-DF.
git clone https://github.com/tensorflow/decision-forests.git
cd decision-forests
```

*Optional:* TensorFlow Decision Forests depends on
[Yggdrasil Decision Forests](https://github.com/google/yggdrasil-decision-forests)
. If you want to edit the Yggdrasil code, you can clone the Yggdrasil repository
and change the path accordingly in
`third_party/yggdrasil_decision_forests/workspace.bzl`.

Compile and run the unit tests of TF-DF with the following command. Note that
`test_bazel.sh` is configured for the default compiler on your machine. Edit the
file directly to change this configuration.

```shell
# Build and test TF-DF.
RUN_TESTS=1 PY_VERSION=3.9 TF_VERSION=2.16.1 ./tools/test_bazel.sh
```

Create and test a pip package with the following command. Replace python3.9 by
the version of python you want to use. Note that you don't have to use the same
version of Python as in the `test_bazel.sh` script.

If your configuration is compatible with
[manylinux2014](https://www.python.org/dev/peps/pep-0571/), a `manylinux2014`
compatible pip package will be produced.

If your configuration is not compatible with manylinux2014, a non
`manylinux2014` compatible pip package will be produced, and the final check
will fail. It does not matter if you want to use TF-DF on your own machine. An
easy way to make the build manylinux2014 compatible is to use the docker
mentioned above.

```shell
# Build and test a Pip package.
./tools/build_pip_package.sh python3.9
```

This command will install the TF-DF pip package and run the example in
`examples/minimal.py`. The Pip package is located in the `dist/` directory.

If you want to create a Pip package for the other compatible version of Python,
run:

```shell
# Install the other versions of python (assume only python3.9 is installed; this is the case in the build docker).
sudo apt-get update && sudo apt-get install python3.9 python3-pip

# Create the Pip package for the other version of python
./tools/build_pip_package.sh python3.9
```

**Alternatively**, you can create the pip package for all the compatible version
of python using pyenv by running the following command. See the header of
`tools/build_pip_package.sh` for more details.

```shell
# Build and test all the Pip package using Pyenv.
./tools/build_pip_package.sh ALL_VERSIONS
```

### MacOS

#### Setup

**Requirements**

-   XCode command line tools
-   Bazel (recommended [Bazelisk](https://github.com/bazelbuild/bazelisk))
-   Homebrew packages: GNU coreutils, GNU sed, GNU grep
-   Pyenv (for building the Pip packages with multiple Python versions)

#### Arm64 CPU

For MacOS systems with ARM64 CPU, follow these steps:

1.  Prepare your environment

    ```shell
    git clone https://github.com/tensorflow/decision-forests.git
    python3 -m venv venv
    source venv/bin/activate
    ```

1.  Decide which Python version and TensorFlow version you want to use and run

    ```shell
    cd decision-forests
    bazel clean --expunge            # Remove old builds (esp. cross-compiled).
    export RUN_TESTS=1               # Whether to run tests after build.
    export PY_VERSION=3.9            # Python version to use for build.
    # TensorFlow version to compile against. This must match exactly the version
    # of TensorFlow used at runtime, otherwise TF-DF may crash unexpectedly.
    export TF_VERSION=2.16.1
    ./tools/test_bazel.sh            # Takes ~15 minutes on a modern Mac.
    ```

1.  Package the build.

    ```shell
    # Building the packages uses different virtualenvs through Pyenv.
    deactivate
    # Build the packages.
    ./tools/build_pip_package.sh ALL_VERSIONS
    ```

1.  The packages can be found in `decision-forests/dist/`.

#### Cross-compiling for Intel CPUs

If you have a MacOS machine with Apple CPU, cross-compile TF-DF for MacOS
machines with Intel CPUs as follows.

1.  Prepare your environment

    ```shell
    git clone https://github.com/tensorflow/decision-forests.git
    python3 -m venv venv
    source venv/source/activate
    ```

1.  Decide which Python version you want to use and run

    ```shell
    cd decision-forests
    bazel clean --expunge            # Remove old builds (esp. cross-compiled).
    export RUN_TESTS=0               # Cross-compiled builds can't run tests.
    export PY_VERSION=3.9            # Python version to use for build.
    # TensorFlow version to compile against. This must match exactly the version
    # of TensorFlow used at runtime, otherwise TF-DF may crash unexpectedly.
    export TF_VERSION=2.16.1
    export MAC_INTEL_CROSSCOMPILE=1  # Enable cross-compilation.
    ./tools/test_bazel.sh            # Takes ~15 minutes on a modern Mac.
    ```

1.  Package the build.

    ```shell
    # Building the packages uses different virtualenvs through Pyenv.
    deactivate
    # Build the packages.
    ./tools/build_pip_package.sh ALL_VERSIONS_MAC_INTEL_CROSSCOMPILE
    ```

1.  The packages can be found in `decision-forests/dist/`. Note that they have
    not been tested and it would be prudent to test them before distribution.

### Windows

A Windows build has been successfully produced in the past, but is not
maintained at this point. See `tools/test_bazel.bat` and `tools/test_bazel.sh`
for (possibly outdated) pointers for compiling on Windows.

For Windows users, [YDF](https://ydf.readthedocs.io) offers official Windows
builds and most of the functionality (and more!) of TF-DF.
