# TensorFlow Decision Forests and TensorFlow Serving

<!-- docs_infra:strip_begin -->

## Table of Contents

<!--ts-->

<!--te-->

<!-- docs_infra:strip_end -->

## Introduction

[TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) is a serving
system for TensorFlow models in production environments. The TF-Serving team
publishes a
[pre-compiled release](https://www.tensorflow.org/tfx/serving/docker) containing
only Core TensorFlow ops.

TensorFlow Decision Forests (TF-DF) models use custom Ops for inference.
Therefore, they are not compatible with the pre-compiled releases of TF-Serving.
The solution to serve TF-DF models with TF-Serving is to re-compile TF-Serving
with TF-DF ops. This document explains how to do so.

We are currently not offering pre-compiled docker images of TF-Serving for TF-DF
on [dockerhub/tensorflow](https://hub.docker.com/u/tensorflow).

**Note:** The
[TF-Serving guide for custom ops](https://www.tensorflow.org/tfx/serving/custom_op)
proposes to copy custom op implementations inside of the TF-Serving repository.
While possible, we won't follow this solution. Instead, we will create a Bazel
dependency to TF-DF. This way, when a new version of TF-DF is released, you can
update your custom TF-Serving easily.

## Instructions

This guide is inspired from the TF-Serving
[compile from source guide](https://www.tensorflow.org/tfx/serving/setup#building_from_source)
. In case of an issue with TF-Serving or Docker, refer to this guide.

### Install Docker

General installation instructions are
[on the Docker site](https://docs.docker.com/get-docker/).

### Clone the build script

Download the source code of TF-Serving.

```shell
git clone https://github.com/tensorflow/serving.git
cd serving
```

### Compile TF-Serving

First, make sure that TF-Serving compiles.

```shell
tools/run_in_docker.sh bazel build -c opt tensorflow_serving/...
```

This command compiles TensorFlow locally. This operation takes a long time (e.g.
~6h on a good machine without distributed compilation). As of today (Nov. 2021),
the compilation of TF-Serving has ~30'000 bazel steps (see the compilation logs)
.

Make sure to enable the
[available instruction sets](https://www.tensorflow.org/tfx/serving/setup#optimized_build)
compatible with your serving environment. TF-DF inference benefit from SIMD
instructions.

### Import TF-DF in the TF-Serving project.

Add the following lines in the `WORKSPACE` file located in the root directory of
TF-Serving. The content should be placed near the top bellow `workspace(name =
"tf_serving")`:

```python
# Import Yggdrasil Decision Forests.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name="ydf",
    urls=[
        "https://github.com/google/yggdrasil-decision-forests/archive/refs/tags/0.2.0.zip"],
    strip_prefix="yggdrasil-decision-forests-0.2.0",
)

# Load the YDF dependencies. However, skip the ones already injected by
# TensorFlow.
load("@ydf//yggdrasil_decision_forests:library.bzl",
     ydf_load_deps="load_dependencies")
ydf_load_deps(
    exclude_repo=[
        "absl",
        "protobuf",
        "zlib",
        "farmhash",
        "gtest",
        "tensorflow",
        "grpc"
    ],
    repo_name="@ydf",
)

# Import TensorFlow Decision Forests.
load("//tensorflow_serving:repo.bzl", "tensorflow_http_archive")
http_archive(
    name="org_tensorflow_decision_forests",
    urls=[
        "https://github.com/tensorflow/decision-forests/archive/refs/tags/0.2.0.zip"],
    strip_prefix="decision-forests-0.2.0",
)
```

At the time of writing this guide, the current version of
[TensorFlow Decision Forests](https://github.com/tensorflow/decision-forests)
and
[Yggdrasil Decision Forests](https://github.com/google/yggdrasil-decision-forests)
are both `0.2.0`. Visit the respective github repositories to check for new
stable releases. You might also have to select version compatible with the
current version of TensorFlow (see
[TF-DF compatibility table](https://github.com/tensorflow/decision-forests/blob/main/documentation/known_issues.md#compatibility-table))
.

### Register the TF-DF custom op.

Add the TF-DF inference ops to the `SUPPORTED_TENSORFLOW_OPS` variable defined
in the file `tensorflow_serving/model_servers/BUILD`.

```python
SUPPORTED_TENSORFLOW_OPS = if_v2([]) + if_not_v2([
]) + [
   # ... other ops
   "@org_tensorflow_decision_forests//tensorflow_decision_forests/tensorflow/ops/inference:kernel_and_op",
   # TF-DF inference op.
]
```

### Recompile TF-Serving

Recompile TF-Serving (similarly as in the "Compile TF-Serving" section above)
with the following flags:

```shell
--define use_tensorflow_io=1 # Use TensorFlow for IO operations.
--define no_absl_statusor=1 # Do not use absl for status (tf uses an old version of absl).
```

The full command might look as follows:

```shell
tools/run_in_docker.sh -d tensorflow/serving:latest-devel bazel \
    build -c opt tensorflow_serving/... \
    --define use_tensorflow_io=1 \
    --define no_absl_statusor=1
```

### Testing the TF-Serving + TF-DF build

In this example, we trained a model on the adult dataset following
[this example](https://github.com/tensorflow/decision-forests/blob/main/examples/minimal.py).

Start TF-Serving on this model:

```shell
MODEL_PATH=/tmp/my_saved_model
MODEL_NAME=my_model

# Make sure that MODEL_PATH contains a version serving sub-directory. For example, the structure should be:
tree $MODEL_PATH
# /path/to/tf-df/model
# └── 1
#     ├── assets
#     ├── keras_metadata.pb
#     ├── saved_model.pb
#     └── variables

bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server \
    --rest_api_port=8501 --model_name=${MODEL_NAME} --model_base_path=${MODEL_PATH}
```

Alternatively, you can start TF-Serving from withing the Docker instance:

```shell
tools/run_in_docker.sh -d tensorflow/serving:latest-devel \
  -o "-p 8501:8501 --mount type=bind,source=${MODEL_PATH},target=/my_model" \
  bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server \
  --rest_api_port=8501 --model_name=${MODEL_NAME} --model_base_path=/my_model
```

Finally, send test requests to the model:

```shell
curl http://localhost:8501/v1/models/${MODEL_NAME}:predict -X POST \
    -d '{"instances": [{"age":[39],"workclass":["State-gov"],"fnlwgt":[77516],"education":["Bachelors"],"education_num":[13],"marital_status":["Never-married"],"occupation":["Adm-clerical"],"relationship":["Not-in-family"],"race":["White"],"sex":["Male"],"capital_gain":[2174],"capital_loss":[0],"hours_per_week":[40],"native_country":["United-States"]}]}'
```
