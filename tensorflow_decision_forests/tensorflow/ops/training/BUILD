load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
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
    data = [":training.so"],
    deps = [
        "//tensorflow_decision_forests/tensorflow:check_version",
        "@release_or_nightly//:tensorflow_pkg",
    ],
)

cc_binary(
    name = "training.so",
    linkshared = 1,
    deps = [":kernel_and_op"],
)

# Op+kernel
cc_library(
    name = "kernel_and_op",
    deps = [
        ":kernel",
        ":op",
        "//tensorflow_decision_forests/tensorflow:canonical_learners",
        "//tensorflow_decision_forests/tensorflow:dataset_formats",
        "//tensorflow_decision_forests/tensorflow:distribution_engines",
    ],
    alwayslink = 1,
)

# Declaration of the ops.
cc_library(
    name = "op",
    srcs = [
        "op.cc",
        "op_grpc_worker.cc",
    ],
    linkstatic = 1,
    deps = [] + select({
        "@ydf//yggdrasil_decision_forests:tensorflow_with_header_lib": [
            "@release_or_nightly//:tensorflow_libtensorflow_framework",
            "@release_or_nightly//:tensorflow_tf_header_lib",
        ],
        "//conditions:default": ["@org_tensorflow//tensorflow/core:framework"],
    }),
    alwayslink = 1,
)

# Definition of the ops i.e. the kernels.
cc_library(
    name = "kernel",
    srcs = [
        "feature_on_file.cc",
        "kernel.cc",
        "kernel_grpc_worker.cc",
        "kernel_long_process.cc",
        "kernel_on_file.cc",
    ],
    hdrs = [
        "feature_on_file.h",
        "features.h",
        "kernel.h",
    ],
    defines = select({
        "//tensorflow_decision_forests:stop_training_on_interrupt": ["TFDF_STOP_TRAINING_ON_INTERRUPT"],
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_inference",
        "@ydf//yggdrasil_decision_forests/dataset:vertical_dataset",
        "@ydf//yggdrasil_decision_forests/learner:abstract_learner",
        "@ydf//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "@ydf//yggdrasil_decision_forests/learner:learner_library",
        "@ydf//yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache:column_cache",
        "@ydf//yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache:dataset_cache_cc_proto",
        "@ydf//yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache:dataset_cache_common",
        "@ydf//yggdrasil_decision_forests/model:abstract_model",
        "@ydf//yggdrasil_decision_forests/model:model_library",
        "@ydf//yggdrasil_decision_forests/model/decision_tree:decision_forest_interface",
        "@ydf//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "@ydf//yggdrasil_decision_forests/model/random_forest",
        "@ydf//yggdrasil_decision_forests/utils:concurrency",
        "@ydf//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "@ydf//yggdrasil_decision_forests/utils:filesystem_tensorflow_impl",  # keep
        "@ydf//yggdrasil_decision_forests/utils:logging",
        "@ydf//yggdrasil_decision_forests/utils:status_macros",
        "@ydf//yggdrasil_decision_forests/utils:synchronization_primitives",
        "@ydf//yggdrasil_decision_forests/utils/distribute/implementations/grpc:grpc_manager",
        "@ydf//yggdrasil_decision_forests/utils/distribute/implementations/grpc:grpc_worker",
    ] + select({
        "@ydf//yggdrasil_decision_forests:tensorflow_with_header_lib": [
            "@release_or_nightly//:tensorflow_libtensorflow_framework",
            "@release_or_nightly//:tensorflow_tf_header_lib",
        ],
        "//conditions:default": [
            "@org_tensorflow//tensorflow/core:framework",
            "@org_tensorflow//tensorflow/core/platform:path",
        ],
    }),
    alwayslink = 1,
)

# Tests
# =====

py_test(
    name = "op_test",
    srcs = ["op_test.py"],
    deps = [
        ":op_py",
        "@pypi_absl_py//:pkg",  # /logging
        "@release_or_nightly//:tensorflow_pkg",
    ],
)
