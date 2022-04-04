# TensorFlow Decision Forests and TensorFlow Serving

<!-- docs_infra:strip_begin -->

## Table of Contents

<!--ts-->

<!--te-->

<!-- docs_infra:strip_end -->

## TL;DR

To run TF Decision Forests models in TF-Serving, use the precompiled
[TF-Sering+TF-Decision Forests package](https://github.com/tensorflow/decision-forests/releases)
**(recommended)**, or compile it from source
([instructions](#compile-tf-seringtf-decision-forests-from-source),
[automated building](https://github.com/tensorflow/decision-forests/tree/main/tools/tf_serving)).

Run
[the TF-Serving+TF-DF example](https://github.com/tensorflow/decision-forests/tree/main/tools/tf_serving)
for a demonstration of TF-Serving + TF-DF.

## Introduction

[TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) is a system
to run TensorFlow models in production environments. More precisely, TensorFlow
Serving is a binary that expose model predictions through gRPC and HTTP. The
TF-Serving team publishes a
[pre-compiled release](https://www.tensorflow.org/tfx/serving/docker) compatible
with models only containing *canonical* TensorFlow ops.

The TensorFlow Decision Forests (TF-DF) library uses *custom* TensorFlow Ops for
inference. Therefore, TF-DF models are not compatible with the pre-compiled
releases of TF-Serving. If you try, the following error will be raised:

*NOT_FOUND: Op type not registered 'SimpleMLCreateModelResource' in binary
running on gbm1.zrh.corp.google.com. Make sure the Op and Kernel are registered
in the binary running in this process. Note that if you are loading a saved
graph which used ops from tf.contrib, accessing (e.g.) `tf.contrib.resampler`
should be done before importing the graph, as contrib ops are lazily registered
when the module is first accessed.*

Two options are available to run TF-DF in TF Serving:

1.  Use the pre-compiled version of TF Serving managed by the TF-DF team and
    compatible with TF-DF. TF Serving binaries are related on the
    [TF-DF GitHub release page](https://github.com/tensorflow/decision-forests/releases)
    . For example, search for the latest `tf_serving_linux.zip`.
1.  Compile TF Serving from source with support for TF-DF using the instructions
    below.

**Remarks:**

-   [TF-DF TF-Serving compile script](https://github.com/tensorflow/decision-forests/tree/main/tools/tf_serving)
    is an experimental solution to compile TF-Serving in TF-DF automatically. It
    is equivalent to the instruction below.

-   "TF-Serving + TF-Decision Forests" runs independently the Python
    installation of TF-Decision Forests.

-   TF-Decision Forests models are backward compatible: For example, a model
    trained with TF-DF v0.3 can be run with TF-DF v0.4.

-   In the vast majority of cases, TF-DF models are foward compatible: For
    example, a model trained with TF-DF v0.4 can be run with the TF-DF v0.1.

## Compile TF-Sering+TF-Decision Forests from source manually

### Troubleshooting

For errors specific to TF Serving, refer to the
[TF-Serving compilation guide](https://www.tensorflow.org/tfx/serving/setup#building_from_source)
.

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
TF-Serving. The content should be placed near the top below `workspace(name =
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

Run
[the TF-Serving+TF-DF example](https://github.com/tensorflow/decision-forests/tree/main/tools/tf_serving)
to test to test your binary.
