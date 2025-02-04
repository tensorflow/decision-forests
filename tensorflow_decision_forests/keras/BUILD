load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("//tensorflow_decision_forests/keras/wrapper:wrapper.bzl", "py_wrap_yggdrasil_learners")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

cc_library(
    name = "learners",
    deps = ["//tensorflow_decision_forests/tensorflow:canonical_learners"],
    alwayslink = 1,
)

py_library(
    name = "keras",
    srcs = ["__init__.py"],
    deps = [
        ":yggdrasil_decision_forests_protos",
        ":core",
        ":wrappers",
        # ":wrappers_pre_generated" # Pre-generated alternative to ":wrappers".
    ],
)

# Yggdrasil Decision Forests protos included in the package.
py_library(
    name = "yggdrasil_decision_forests_protos",
    deps = [
        "@ydf//yggdrasil_decision_forests/learner/decision_tree:decision_tree_py_proto",
        "@ydf//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_py_proto",
        "@ydf//yggdrasil_decision_forests/learner/random_forest:random_forest_py_proto",
    ],
)

py_wrap_yggdrasil_learners(
    name = "wrappers",
    learner_deps = [":learners"],
)

# Pre-generated result of ":wrappers".
#
# bazel build -c opt //tensorflow_decision_forests/keras:wrappers
# bazel-bin/third_party/tensorflow_decision_forests/keras/wrappers_wrapper_main > third_party/tensorflow_decision_forests/keras/wrappers_pre_generated.py
py_library(
    name = "wrappers_pre_generated",
    srcs = ["wrappers_pre_generated.py"],
    deps = [
        ":core",
        "//tensorflow_decision_forests/component/tuner",
        "@release_or_nightly//:tensorflow_pkg",
        "@release_or_nightly//:tf_keras_pkg",
        "@ydf//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "@ydf//yggdrasil_decision_forests/model:abstract_model_py_proto",
    ],
)

py_library(
    name = "core_inference",
    srcs = ["core_inference.py"],
    tags = [
        # Certain deps are not provided on purpose, so this cannot be a python_strict_library.
        "ignore_for_dep=third_party.tensorflow.python.framework.type_spec_registry",
    ],
    deps = [
        ":keras_internal",
        "@release_or_nightly//:tensorflow_pkg",
        "@release_or_nightly//:tf_keras_pkg",
        # TensorFlow /distribute:input_lib,
        "//tensorflow_decision_forests/component/inspector",
        "//tensorflow_decision_forests/component/tuner",
        "//tensorflow_decision_forests/tensorflow:core_inference",
        "//tensorflow_decision_forests/tensorflow:tf_logging",
        "//tensorflow_decision_forests/tensorflow/ops/inference:api_py",
        "@ydf//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "@ydf//yggdrasil_decision_forests/learner/multitasker:multitasker_py_proto",
        "@ydf//yggdrasil_decision_forests/model:abstract_model_py_proto",
        "@ydf//yggdrasil_decision_forests/utils/distribute/implementations/grpc:grpc_py_proto",
    ],
)

py_library(
    name = "core",
    srcs = ["core.py"],
    deps = [
        ":core_inference",
        ":keras_internal",
        "@release_or_nightly//:tensorflow_pkg",
        "@release_or_nightly//:tf_keras_pkg",
        # TensorFlow /data/ops:dataset_ops,
        "//tensorflow_decision_forests/component/inspector",
        "//tensorflow_decision_forests/component/tuner",
        "//tensorflow_decision_forests/tensorflow:cc_logging",
        "//tensorflow_decision_forests/tensorflow:core",
        "//tensorflow_decision_forests/tensorflow:tf1_compatibility",
        "//tensorflow_decision_forests/tensorflow:tf_logging",
        "//tensorflow_decision_forests/tensorflow/ops/inference:api_py",
        "//tensorflow_decision_forests/tensorflow/ops/training:op_py",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "@ydf//yggdrasil_decision_forests/model:abstract_model_py_proto",
        "@ydf//yggdrasil_decision_forests/utils/distribute/implementations/grpc:grpc_py_proto",
    ],
)

py_library(
    name = "keras_internal",
    srcs = ["keras_internal.py"],
    deps = [
        "@release_or_nightly//:tf_keras_pkg",
        # tf_keras/engine dep,
        # tf_keras/engine:data_adapter dep,
        # tf_keras/engine:input_layer dep,
        # tf_keras/feature_column:dense_features_v2 dep,
        ## tf_keras/layers dep,
        # tf_keras/utils:dataset_creator dep,
    ],
)

# Tests
# =====

py_test(
    name = "keras_test",
    size = "large",
    srcs = ["keras_test.py"],
    data = [
        ":synthetic_dataset",
        ":test_runner",
        "//tensorflow_decision_forests/test_data",
        "@ydf//yggdrasil_decision_forests/test_data",
    ],
    shard_count = 50,
    deps = [
        ":core",
        ":keras",
        ":keras_internal",
        "@com_google_protobuf//:python_srcs",
        # absl/flags dep,
        # absl/logging dep,
        # absl/testing:parameterized dep,
        "@pypi_numpy//:pkg",
        "@pypi_pandas//:pkg",
        "@release_or_nightly//:tensorflow_pkg",
        "@release_or_nightly//:tf_keras_pkg",
        "//tensorflow_decision_forests/component/inspector",
        "//tensorflow_decision_forests/component/model_plotter",
        "//tensorflow_decision_forests/tensorflow:core",
        "@ydf//yggdrasil_decision_forests/dataset:synthetic_dataset_py_proto",
        "@ydf//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "@ydf//yggdrasil_decision_forests/learner/decision_tree:decision_tree_py_proto",
        "@ydf//yggdrasil_decision_forests/learner/random_forest:random_forest_py_proto",
        "@ydf//yggdrasil_decision_forests/model:abstract_model_py_proto",
    ],
)

py_test(
    name = "keras_tuner_test",
    size = "large",
    srcs = ["keras_tuner_test.py"],
    data = ["@ydf//yggdrasil_decision_forests/test_data"],
    shard_count = 2,
    deps = [
        ":core",
        ":keras",
        "@com_google_protobuf//:python_srcs",
        # absl/flags dep,
        # absl/logging dep,
        # absl/testing:parameterized dep,
        "@pypi_numpy//:pkg",
        "@pypi_pandas//:pkg",
        "@release_or_nightly//:tensorflow_pkg",
        "//tensorflow_decision_forests",
        "@release_or_nightly//:tf_keras_pkg",
    ],
)

# This test relies on the support of TF PS distribution strategy and TF-DF.
# Note: TF PS distribution strategy and TF-DF are currently not compatible in non-monolithic build of TensorFlow+TFDF (e.g. OSS TFDF).
#
# This test is expected to fail TF PS distributed training is disabled (i.e.
# enabling the ":disable_tf_ps_distribution_strategy" rule).
py_test(
    name = "keras_distributed_test",
    size = "large",
    srcs = ["keras_distributed_test.py"],
    data = [
        ":grpc_worker_main",
        "@ydf//yggdrasil_decision_forests/test_data",
    ],
    shard_count = 10,
    tags = [
        # This test is notoriously flaky.
        "manual",
    ],
    deps = [
        ":keras_internal",
        # absl/flags dep,
        # absl/logging dep,
        # absl/testing:parameterized dep,
        "@pypi_numpy//:pkg",
        "@pypi_pandas//:pkg",
        "@pypi_portpicker//:pkg",
        "@release_or_nightly//:tensorflow_pkg",
        # TensorFlow /distribute:distribute_lib,
        "//tensorflow_decision_forests",
        "@ydf//yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees:dgbt_py_proto",
    ],
)

py_binary(
    name = "test_runner",
    srcs = ["test_runner.py"],
    data = ["@ydf//yggdrasil_decision_forests/test_data"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        # absl/logging dep,
        "@pypi_numpy//:pkg",
        "@pypi_pandas//:pkg",
        "@release_or_nightly//:tensorflow_pkg",
        "@release_or_nightly//:tf_keras_pkg",
        "//tensorflow_decision_forests/tensorflow/ops/inference:op_py",
    ],
)

tf_cc_binary(
    name = "synthetic_dataset",
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@ydf//yggdrasil_decision_forests/cli/utils:synthetic_dataset_lib_with_main",
    ],
)

tf_cc_binary(
    name = "grpc_worker_main",
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@ydf//yggdrasil_decision_forests/utils/distribute/implementations/grpc:grpc_worker_lib_with_main",
    ],
)

py_test(
    name = "keras_internal_test",
    srcs = ["keras_internal_test.py"],
    deps = [
        ":keras_internal",
        "@release_or_nightly//:tensorflow_pkg",
    ],
)
