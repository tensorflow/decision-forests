# Installation

<!-- docs_infra:strip_begin -->

## Table of Contents

<!--ts-->

*   [Installation](#installation)
    *   [Table of Contents](#table-of-contents)
    *   [Installation with Pip](#installation-with-pip)
    *   [Build from source](#build-from-source)
        *   [Linux](#linux)
            *   [Setup](#setup)
            *   [Compilation](#compilation)
        *   [MacOS](#macos)
            *   [Setup](#setup-1)
            *   [Compilation](#compilation-1)
    *   [Final note](#final-note)
    *   [Troubleshooting](#troubleshooting)

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

### Linux

#### Setup

**Requirements**

-   Bazel >= 3.7.2
-   Python >= 3
-   Git
-   Python packages: numpy tensorflow pandas

Instead of installing the dependencies by hands, you can use the
[TensorFlow Build docker](https://github.com/tensorflow/build). If you choose
this options, install Docker:

- [Docker](https://docs.docker.com/get-docker/).

#### Compilation

Download TensorFlow Decision Forests as follows:

```shell
# Download the source code of TF-DF.
git clone https://github.com/tensorflow/decision-forests.git
cd decision-forests
```

**Optional:** TensorFlow Decision Forests depends on
[Yggdrasil Decision Forests](https://github.com/google/yggdrasil-decision-forests)
. If you want to edit the Yggdrasil code, you can clone the Yggdrasil github and
change the path accordingly in
`third_party/yggdrasil_decision_forests/workspace.bzl`.

**Optional:** If you want to use the docker option, run the
`start_compile_docker.sh` script and continue to the next step. If you don't
want to use the docker option, continue to the next step directly.

```shell
# Optional: Install and start the build docker.
/tools/start_compile_docker.sh
```

Compile and run the unit tests of TF-DF with the following command. Note
that `test_bazel.sh` is configured for `python3.8` and the default compiler on
your machine. Edit the file directly to change this configuration.

```shell
# Build and test TF-DF.
./tools/test_bazel.sh
```

Create and test a pip package with the following command. Replace python3.8 by
the version of python you want to use. Note that you don't have to use the same
version of Python as in the `test_bazel.sh` script.

If your configuration is compatible with
[manylinux2010](https://www.python.org/dev/peps/pep-0571/), a `manylinux2010`
compatible pip package will be produced.

If your configuration is not compatible with manylinux2010, a non
`manylinux2010` compatible pip package will be produced, and the final check
will fail. It does not matter if you want to use TF-DF on your own machine. An
easy way to make the build manylinux2010 compatible is to use the docker
mentioned above.

```shell
# Build and test a Pip package.
./tools/build_pip_package.sh python3.8
```

This command will install the TF-DF pip package and run the example in
`examples/minimal.py`. The Pip package is located in the `dist/` directory.

If you want to create a Pip package for the other compatible version of Python,
run:

```shell
# Install the other versions of python (assume only python3.8 is installed; this is the case in the build docker).
sudo apt-get update && sudo apt-get install python3.7 python3.9 python3-pip

# Create the Pip package for the other version of python
./tools/build_pip_package.sh python3.7
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

-   Coreutils (tested with `brew install coreutils`)
-   Bazel >= 3.7.2
-   Python >= 3 (tested with `brew install python`)
-   Git
-   JDK 11
-   Python packages: numpy tensorflow pandas

#### Compilation

Follow the same steps as for the linux compilation without Docker.

Note: Currently, multi-threading support is broken on MacOS.

## Final note

Compiling TF-DF relies (since 17th Dec. 2021) on the TensorFlow Pip package and
the TensorFlow Bazel dependency. A small part of TensorFlow will be compiled.
Compiling TF-DF on a single powerful workstation takes ~10 minutes.

## Troubleshooting

**Note:** Check also Yggdrasil's
[Troubleshooting](https://github.com/google/yggdrasil-decision-forests/blob/main/documentation/installation.md#troubleshooting)
page.
