# Implementation of the Yggdrasil Distribute API using TensorFlow Distribution Strategies.
#

load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_gen_op_wrapper_py")
load("@ydf//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library")

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
    deps = [
        ":op_py",
        # absl:app dep,
        # absl/flags dep,
        # absl/logging dep,
        # TensorFlow Python,
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

# TF Dependencies that are not exported in the Pip package.
TF_CC_API_DEP = [
    # E.g. tensorflow/cc/client/client_session.h
    "@org_tensorflow//tensorflow/cc:client_session",
    # E.g. org_tensorflow/tensorflow/cc/ops/array_ops.h
    "@org_tensorflow//tensorflow/cc:cc_ops",
]

cc_library(
    name = "tf_distribution",
    srcs = ["tf_distribution.cc"],
    hdrs = ["tf_distribution.h"],
    deps = [
        ":tf_distribution_common",
        ":op",
        ":tf_distribution_cc_proto",
        "@ydf//yggdrasil_decision_forests/utils:synchronization_primitives",
        "@com_github_tencent_rapidjson//:rapidjson",
        "@ydf//yggdrasil_decision_forests/utils:concurrency",
        "@ydf//yggdrasil_decision_forests/utils:filesystem",
        "@ydf//yggdrasil_decision_forests/utils:logging",
        "@ydf//yggdrasil_decision_forests/utils:status_macros",
        "@ydf//yggdrasil_decision_forests/utils:tensorflow",
        "@ydf//yggdrasil_decision_forests/utils:uid",
        "@ydf//yggdrasil_decision_forests/utils/distribute:core",
        "@ydf//yggdrasil_decision_forests/utils/distribute:distribute_cc_proto",
        "@ydf//yggdrasil_decision_forests/utils/distribute:utils",
    ] + select({
        "@ydf//yggdrasil_decision_forests:tensorflow_with_header_lib": [
            "@tensorflow_pypi//:libtensorflow_framework",
            "@tensorflow_pypi//:tf_header_lib",
        ] + TF_CC_API_DEP,
        "//conditions:default": [
            "@org_tensorflow//tensorflow/core:framework",
            "@org_tensorflow//tensorflow/cc:cc_ops",
            "@org_tensorflow//tensorflow/cc:client_session",
            "@org_tensorflow//tensorflow/cc:ops",
            "@org_tensorflow//tensorflow/cc:scope",
            "@org_tensorflow//tensorflow/core:protos_all_cc",
            "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:grpc_session",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "tf_distribution_common",
    srcs = ["tf_distribution_common.cc"],
    hdrs = ["tf_distribution_common.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        # TODO(gbm): Usepayload on next TF release (introduced on 2021-11-02).
        # "@org_tensorflow//tensorflow/core/distributed_runtime:error_payloads",
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
    data = [":distribute.so"],
    srcs_version = "PY3",
    deps = [
        # TensorFlow Python,
    ],
)

cc_binary(
    name = "distribute.so",
    linkshared = 1,
    deps = [":kernel_and_op"],
)

# Op+kernel
cc_library(
    name = "kernel_and_op",
    deps = [
        ":kernel",
        ":op",
    ],
    alwayslink = 1,
)

# Declaration of the ops.
cc_library(
    name = "op",
    srcs = ["tf_distribution_op.cc"],
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
    srcs = ["tf_distribution_kernel.cc"],
    deps = [
        ":tf_distribution_common",
        ":tf_distribution_cc_proto",
        "@com_google_absl//absl/status",
        "@ydf//yggdrasil_decision_forests/utils:concurrency",
        "@ydf//yggdrasil_decision_forests/utils:status_macros",
        "@ydf//yggdrasil_decision_forests/utils:tensorflow",
        "@ydf//yggdrasil_decision_forests/utils/distribute:all_workers",
        "@ydf//yggdrasil_decision_forests/utils/distribute:core",
        "@ydf//yggdrasil_decision_forests/utils/distribute:distribute_cc_proto",
        "@ydf//yggdrasil_decision_forests/utils/distribute:utils",
    ] + select({
        "@ydf//yggdrasil_decision_forests:tensorflow_with_header_lib": [
            "@tensorflow_pypi//:libtensorflow_framework",
            "@tensorflow_pypi//:tf_header_lib",
        ] + TF_CC_API_DEP,
        "//conditions:default": [
            "@org_tensorflow//tensorflow/core:framework",
            "@org_tensorflow//tensorflow/cc:cc_ops",
            "@org_tensorflow//tensorflow/cc:client_session",
            "@org_tensorflow//tensorflow/cc:ops",
            "@org_tensorflow//tensorflow/cc:scope",
            "@org_tensorflow//tensorflow/core:protos_all_cc",
            "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:grpc_session",
        ],
    }),
    alwayslink = 1,
)

# Tests
# =====

tf_cc_test(
    name = "tf_distribution_test",
    timeout = "long",
    srcs = ["tf_distribution_test.cc"],
    data = [
        ":tf_distribution_py_worker",
    ],
    shard_count = 7,
    tags = [
        "noasan",  # OOM as a TF is initialized in each workers.
    ],
    deps = [
        ":tf_distribution",
        ":tf_distribution_cc_proto",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core/platform:test",
        "@ydf//yggdrasil_decision_forests/utils:filesystem",
        "@ydf//yggdrasil_decision_forests/utils:test",
        "@ydf//yggdrasil_decision_forests/utils/distribute",
        "@ydf//yggdrasil_decision_forests/utils/distribute:distribute_without_implementations",
        "@ydf//yggdrasil_decision_forests/utils/distribute:test_utils",
        "@ydf//yggdrasil_decision_forests/utils/distribute:toy_worker",
    ],
)
