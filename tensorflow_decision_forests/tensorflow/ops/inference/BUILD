load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

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
    srcs_version = "PY3",
    deps = [
        ":op_py",
        # absl/logging dep,
        # six dep,
        # TensorFlow Python,
        # TensorFlow :framework_ops,
        # TensorFlow /eager:context,
        # TensorFlow /training/tracking,
        # TensorFlow /training/tracking:base,
        "//tensorflow_decision_forests/component/inspector",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf//yggdrasil_decision_forests/model:abstract_model_py_proto",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    srcs_version = "PY3",
    deps = [
        # absl/logging dep,
        # TensorFlow Python,
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
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    deps = [
        # TensorFlow Python,
        "//tensorflow_decision_forests/tensorflow:check_version",
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
cc_library(
    name = "op",
    srcs = ["op.cc"],
    linkstatic = 1,
    deps = [] + select({
        "@ydf//yggdrasil_decision_forests:tensorflow_with_header_lib": [
            "@tensorflow_pypi//:libtensorflow_framework",
            "@tensorflow_pypi//:tf_header_lib",
        ],
        "//conditions:default": ["@org_tensorflow//tensorflow/core:framework"],
    }),
    alwayslink = 1,
)

# Definition of the ops i.e. the kernels.
cc_library(
    name = "kernel",
    srcs = ["kernel.cc"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "@ydf//yggdrasil_decision_forests/dataset:vertical_dataset",
        "@ydf//yggdrasil_decision_forests/model:abstract_model",
        "@ydf//yggdrasil_decision_forests/model:model_library",
        "@ydf//yggdrasil_decision_forests/model/decision_tree:decision_forest_interface",
        "@ydf//yggdrasil_decision_forests/utils:compatibility",
        "@ydf//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "@ydf//yggdrasil_decision_forests/utils:status_macros",
        "@ydf//yggdrasil_decision_forests/utils:tensorflow",
    ] + select({
        "@ydf//yggdrasil_decision_forests:tensorflow_with_header_lib": [
            "@tensorflow_pypi//:libtensorflow_framework",
            "@tensorflow_pypi//:tf_header_lib",
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
    python_version = "PY3",
    deps = [
        ":api_py",
        ":test_utils",
        # absl/flags dep,
        # absl/logging dep,
        # absl/testing:parameterized dep,
        # TensorFlow Python,
    ],
)

py_test(
    name = "tf2_test",
    srcs = ["tf2_test.py"],
    data = [
        "@ydf//yggdrasil_decision_forests/test_data",
    ],
    python_version = "PY3",
    deps = [
        ":api_py",
        ":test_utils",
        # absl/flags dep,
        # absl/logging dep,
        # absl/testing:parameterized dep,
        # TensorFlow Python,
    ],
)
