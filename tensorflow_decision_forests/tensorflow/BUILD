load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Canonical models and learners
# =============================

# Also include the inference engines for the models.
cc_library(
    name = "canonical_models",
    deps = [
        "@ydf//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "@ydf//yggdrasil_decision_forests/model/multitasker",
        "@ydf//yggdrasil_decision_forests/model/random_forest",
        "@ydf//yggdrasil_decision_forests/serving/decision_forest:register_engines",
    ],
    alwayslink = 1,
)

# Available learning algorithms.
cc_library(
    name = "canonical_learners",
    deps = [
        "@ydf//yggdrasil_decision_forests/learner/cart",
        "@ydf//yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees:dgbt",
        "@ydf//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "@ydf//yggdrasil_decision_forests/learner/hyperparameters_optimizer",
        "@ydf//yggdrasil_decision_forests/learner/multitasker",
        "@ydf//yggdrasil_decision_forests/learner/random_forest",
    ],
    alwayslink = 1,
)

# Available engines for distributed training.
cc_library(
    name = "distribution_engines",
    deps = [
        "@ydf//yggdrasil_decision_forests/utils/distribute",
        "@ydf//yggdrasil_decision_forests/utils/distribute:all_workers",
    ],
    alwayslink = 1,
)

# Dataset formats supported when training from file
# =================================================

# Dataset format supported by the disk-based trainer.
cc_library(
    name = "dataset_formats",
    deps = [
        "@ydf//yggdrasil_decision_forests/dataset:csv_example_reader",
        "@ydf//yggdrasil_decision_forests/dataset/tensorflow:tf_example_io_tfrecord",
        "@ydf//yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache:dataset_cache_reader",
    ],
    alwayslink = 1,
)

# Bzl Libraries
# =============

bzl_library(
    name = "utils_bzl",
    srcs = ["utils.bzl"],
)

# Libraries
# =========

py_library(
    name = "core_inference",
    srcs = ["core_inference.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_decision_forests/component/inspector",
        "@pypi_numpy//:pkg",
        "@pypi_pandas//:pkg",
        "@release_or_nightly//:tensorflow_pkg",
        "@release_or_nightly//:tf_keras_pkg",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf//yggdrasil_decision_forests/model:abstract_model_py_proto",
    ],
)

py_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY3",
    deps = [
        ":core_inference",
        "@pypi_numpy//:pkg",
        "@pypi_pandas//:pkg",
        "@release_or_nightly//:tensorflow_pkg",
        "@release_or_nightly//:tf_keras_pkg",
        # TensorFlow /distribute:distribute_lib,
        # TensorFlow /distribute:parameter_server_strategy_v2,
        # TensorFlow /distribute/coordinator:cluster_coordinator,
        # TensorFlow /distribute/coordinator:coordinator_context,
        "//tensorflow_decision_forests/component/inspector",
        "//tensorflow_decision_forests/component/py_tree",
        "//tensorflow_decision_forests/tensorflow/ops/training:op_py",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "@ydf//yggdrasil_decision_forests/model:abstract_model_py_proto",
        "@ydf//yggdrasil_decision_forests/model:hyperparameter_py_proto",
        "@ydf//yggdrasil_decision_forests/utils/distribute/implementations/grpc:grpc_py_proto",
    ],
)

py_library(
    name = "check_version",
    srcs = ["check_version.py"],
    srcs_version = "PY3",
    deps = [
        "@release_or_nightly//:tensorflow_pkg",
    ],
)

py_library(
    name = "tf_logging",
    srcs = ["tf_logging.py"],
    srcs_version = "PY3",
    deps = [
        # absl/logging dep,
        "@release_or_nightly//:tensorflow_pkg",
    ],
)

py_library(
    name = "cc_logging",
    srcs = ["cc_logging.py"],
    srcs_version = "PY3",
    deps = [
        ":tf_logging",
        "//tensorflow_decision_forests/tensorflow/ops/training:op_py",
        "@release_or_nightly//:tensorflow_pkg",
    ],
)

py_library(
    name = "tf1_compatibility",
    srcs = ["tf1_compatibility.py"],
    srcs_version = "PY3",
    deps = [
        "@release_or_nightly//:tensorflow_pkg",
        "@release_or_nightly//:tf_keras_pkg",
        # TensorFlow /trackable,
    ],
)

# Tests
# =====

py_test(
    name = "core_test",
    srcs = ["core_test.py"],
    data = ["@ydf//yggdrasil_decision_forests/test_data"],
    python_version = "PY3",
    deps = [
        ":core",
        # absl/logging dep,
        "@pypi_pandas//:pkg",
        "@release_or_nightly//:tensorflow_pkg",
        "@release_or_nightly//:tf_keras_pkg",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf//yggdrasil_decision_forests/model:hyperparameter_py_proto",
    ],
)

py_test(
    name = "check_version_test",
    srcs = ["check_version_test.py"],
    python_version = "PY3",
    deps = [
        ":check_version",
        "@release_or_nightly//:tensorflow_pkg",
    ],
)
