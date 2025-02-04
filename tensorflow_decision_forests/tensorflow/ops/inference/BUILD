load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_gen_op_wrapper_py",
)
load("@rules_python//python:defs.bzl", "py_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

# User-facing library to run models in TensorFlow.
py_library(
    name = "api_py",
    srcs = ["api.py"],
    deps = [
        ":op_py",
        # absl/logging dep,
        "@pypi_six//:pkg",
        "@release_or_nightly//:tensorflow_pkg",
        # TensorFlow /eager:context,
        # TensorFlow /framework:ops,
        "//tensorflow_decision_forests/component/inspector",
        "//tensorflow_decision_forests/tensorflow:tf1_compatibility",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf//yggdrasil_decision_forests/model:abstract_model_py_proto",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = [
        # absl/logging dep,
        "@release_or_nightly//:tensorflow_pkg",
        "//tensorflow_decision_forests/component/inspector:blob_sequence",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf//yggdrasil_decision_forests/model:abstract_model_py_proto",
        "@ydf//yggdrasil_decision_forests/model/decision_tree:decision_tree_py_proto",
        "@ydf//yggdrasil_decision_forests/model/gradient_boosted_trees:gradient_boosted_trees_py_proto",
        "@ydf//yggdrasil_decision_forests/model/random_forest:random_forest_py_proto",
        "@ydf//yggdrasil_decision_forests/utils:distribution_py_proto",
    ],
)

# TF Ops
# ======

py_library(
    name = "op_py",
    srcs = ["op.py"],
    deps = [
    ] + select({
        "@ydf//yggdrasil_decision_forests:tensorflow_with_header_lib": [":op_py_dynamic"],
        "//conditions:default": [":op_py_static"],
    }),
)

tf_gen_op_wrapper_py(
    name = "op_py_static",
    out = "op_static.py",
    deps = [":op"],
)

py_library(
    name = "op_py_dynamic",
    srcs = ["op_dynamic.py"],
    data = [":inference.so"],
    deps = [
        "//tensorflow_decision_forests/tensorflow:check_version",
        "@release_or_nightly//:tensorflow_pkg",
    ],
)

cc_binary(
    name = "inference.so",
    linkshared = 1,
    deps = [":kernel_and_op"],
)

# Op+kernel
cc_library(
    name = "kernel_and_op",
    deps = [
        ":kernel",
        ":op",
        "//tensorflow_decision_forests/tensorflow:canonical_models",
    ],
    alwayslink = 1,
)

# Declaration of the ops.
OP_SRCS = ["op.cc"]

OP_DEPS = select({
    "@ydf//yggdrasil_decision_forests:tensorflow_with_header_lib": [
        "@release_or_nightly//:tensorflow_libtensorflow_framework",
        "@release_or_nightly//:tensorflow_tf_header_lib",
    ],
    "//conditions:default": ["@org_tensorflow//tensorflow/core:framework"],
})

cc_library(
    name = "op",
    srcs = OP_SRCS,
    linkstatic = 1,
    deps = OP_DEPS + [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@ydf//yggdrasil_decision_forests/utils:status_macros",
    ],
    alwayslink = 1,
)

# Definition of the ops i.e. the kernels.
cc_library(
    name = "kernel",
    srcs = ["kernel.cc"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "@ydf//yggdrasil_decision_forests/dataset:vertical_dataset",
        "@ydf//yggdrasil_decision_forests/model:abstract_model",
        "@ydf//yggdrasil_decision_forests/model:model_library",
        "@ydf//yggdrasil_decision_forests/model/decision_tree:decision_forest_interface",
        "@ydf//yggdrasil_decision_forests/utils:compatibility",
        "@ydf//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "@ydf//yggdrasil_decision_forests/utils:status_macros",
    ] + select({
        "@ydf//yggdrasil_decision_forests:tensorflow_with_header_lib": [
            "@release_or_nightly//:tensorflow_libtensorflow_framework",
            "@release_or_nightly//:tensorflow_tf_header_lib",
        ],
        "//conditions:default": ["@org_tensorflow//tensorflow/core:framework"],
    }),
    alwayslink = 1,
)

# Tests
# =====

py_test(
    name = "tf1_test",
    srcs = ["tf1_test.py"],
    data = [
        "@ydf//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":api_py",
        ":test_utils",
        # absl/flags dep,
        # absl/logging dep,
        # absl/testing:parameterized dep,
        "@release_or_nightly//:tensorflow_pkg",
    ],
)

py_test(
    name = "tf2_test",
    srcs = ["tf2_test.py"],
    data = [
        "@ydf//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":api_py",
        ":test_utils",
        # absl/flags dep,
        # absl/logging dep,
        # absl/testing:parameterized dep,
        "@release_or_nightly//:tensorflow_pkg",
    ],
)
