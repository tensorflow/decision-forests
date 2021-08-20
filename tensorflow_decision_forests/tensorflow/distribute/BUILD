# Implementation of the Yggdrasil Distribute API using TensorFlow Distribution Strategies.
#
load("@ydf//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow_decision_forests/tensorflow:utils.bzl", "tf_custom_op_library_external")
load("@org_tensorflow//tensorflow:tensorflow.bzl","tf_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Worker binaries
# ===============

# Worker binary used in TensorFlow/Keras distribution.
# See https://www.tensorflow.org/guide/distributed_training
py_binary(
    name = "tf_distribution_py_worker",
    srcs = ["tf_distribution_py_worker.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    data = [":distribute.so"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        # absl/logging dep,
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

# Proto
# ========

all_proto_library(
    name = "tf_distribution_proto",
    srcs = ["tf_distribution.proto"],
    deps = ["@ydf//yggdrasil_decision_forests/utils/distribute:distribute_proto"],
)

# Library
# =======

cc_library(
    name = "tf_distribution",
    srcs = ["tf_distribution.cc"],
    hdrs = ["tf_distribution.h"],
    deps = [
        ":tf_distribution_cc_proto",
        ":tf_distribution_op",
        "@com_google_absl//absl/synchronization",
        "@com_github_tencent_rapidjson//:rapidjson",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/cc:ops",
        "@org_tensorflow//tensorflow/cc:scope",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:grpc_session",
        "@ydf//yggdrasil_decision_forests/utils:concurrency",
        "@ydf//yggdrasil_decision_forests/utils:filesystem",
        "@ydf//yggdrasil_decision_forests/utils:logging",
        "@ydf//yggdrasil_decision_forests/utils:status_macros",
        "@ydf//yggdrasil_decision_forests/utils:tensorflow",
        "@ydf//yggdrasil_decision_forests/utils/distribute:core",
        "@ydf//yggdrasil_decision_forests/utils/distribute:distribute_cc_proto",
    ],
    alwayslink = 1,
)

# TF Ops
# ======

# Op+kernel for dynamical linking.
tf_custom_op_library_external(
    name = "distribute.so",
    deps = [":register_ops"],
)

cc_library(
    name = "register_ops",
    deps = [
        ":tf_distribution_kernel",
        ":tf_distribution_op",
    ],
    alwayslink = 1,
)

# Declaration of the ops.
cc_library(
    name = "tf_distribution_op",
    srcs = ["tf_distribution_op.cc"],
    linkstatic = 1,
    deps = ["@org_tensorflow//tensorflow/core:framework_headers_lib"],
    alwayslink = 1,
)

# Definition of the ops i.e. the kernels.
cc_library(
    name = "tf_distribution_kernel",
    srcs = ["tf_distribution_kernel.cc"],
    deps = [
        ":tf_distribution_cc_proto",
        "@com_google_absl//absl/status",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@ydf//yggdrasil_decision_forests/utils:concurrency",
        "@ydf//yggdrasil_decision_forests/utils:status_macros",
        "@ydf//yggdrasil_decision_forests/utils:tensorflow",
        "@ydf//yggdrasil_decision_forests/utils/distribute:all_workers",
        "@ydf//yggdrasil_decision_forests/utils/distribute:core",
        "@ydf//yggdrasil_decision_forests/utils/distribute:distribute_cc_proto",
    ],
    alwayslink = 1,
)

# Tests
# =====
