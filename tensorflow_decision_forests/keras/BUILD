load("//tensorflow_decision_forests/keras/wrapper:wrapper.bzl", "py_wrap_yggdrasil_learners")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

cc_library(
    name = "learners",
    deps = [
        "//tensorflow_decision_forests/tensorflow:canonical_learners",
    ],
    alwayslink = 1,
)

py_library(
    name = "keras",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":core",
        ":wrappers",
        # ":wrappers_pre_generated" # Pre-generated alternative to ":wrappers".
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
    srcs_version = "PY3",
    deps = [
        ":core",
        "@org_tensorflow//tensorflow/python",
        "@ydf//yggdrasil_decision_forests/model:abstract_model_py_proto",
    ],
)

py_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY3",
    deps = [
        # absl/logging dep,
        "@org_tensorflow//tensorflow/python",
        "@org_tensorflow//tensorflow/python/training/tracking:base",
        "//tensorflow_decision_forests/component/inspector",
        "//tensorflow_decision_forests/tensorflow:core",
        "//tensorflow_decision_forests/tensorflow/ops/inference:api_py",
        "//tensorflow_decision_forests/tensorflow/ops/training:op_py",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "@ydf//yggdrasil_decision_forests/model:abstract_model_py_proto",
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
        "@ydf//yggdrasil_decision_forests/test_data",
    ],
    python_version = "PY3",
    shard_count = 30,
    deps = [
        ":core",
        ":keras",
        "@com_google_protobuf//:python_srcs",
        # absl/flags dep,
        # absl/logging dep,
        # absl/testing:parameterized dep,
        # enum dep,
        # numpy dep,
        # pandas dep,
        "@org_tensorflow//tensorflow/python",
        "//tensorflow_decision_forests/component/model_plotter",
        "//tensorflow_decision_forests/tensorflow:core",
        "@ydf//yggdrasil_decision_forests/dataset:synthetic_dataset_py_proto",
        "@ydf//yggdrasil_decision_forests/learner/decision_tree:decision_tree_py_proto",
        "@ydf//yggdrasil_decision_forests/learner/random_forest:random_forest_py_proto",
    ],
)

py_binary(
    name = "test_runner",
    srcs = ["test_runner.py"],
    data = ["@ydf//yggdrasil_decision_forests/test_data"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # absl:app dep,
        # absl/flags dep,
        # absl/logging dep,
        # numpy dep,
        # pandas dep,
        "@org_tensorflow//tensorflow/python",
        "//tensorflow_decision_forests/tensorflow/ops/inference:api_py",
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
