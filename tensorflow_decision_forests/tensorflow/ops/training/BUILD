load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow_decision_forests/tensorflow:utils.bzl", "rpath_linkopts_to_tensorflow","tf_custom_op_library_external")
load("//tensorflow_decision_forests/tensorflow:utils.bzl", "rpath_linkopts_to_tensorflow")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# =========

py_library(
    name = "api_py",
    srcs = ["api.py"],
    data = [":training.so"],
    srcs_version = "PY3",
    deps = [
        ":op_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

# TF Ops
# ======

# Op+kernel for dynamical linking.
tf_custom_op_library_external(
    name = "training.so",
    deps = [":kernel_and_op"],
)

# Op+kernel for static linking.
cc_library(
    name = "kernel_and_op",
    deps = [
        ":kernel",
        ":op",
        "//tensorflow_decision_forests/tensorflow:canonical_learners",
    ],
    alwayslink = 1,
)

alias(
    name = "op_py",
    actual = select({
        "//tensorflow_decision_forests:use_precompiled_wrappers": ":op_py_precompile",
        "//conditions:default": ":op_py_no_precompile",
    }),
)

# Python wrapper around the ops.
# This generates a python wrapper around the op.{so,dll}:
# Python wrapper around the ops.
# This is the reason why tf needs to be entirely compiled in OSS setting.
tf_gen_op_wrapper_py(
    name = "op_py_no_precompile",
    out = "op.py",
    cc_linkopts = rpath_linkopts_to_tensorflow("op"),
    deps = [":op"],
)

# Pre-compiled version of "tf_gen_op_wrapper_py" above.
#
# Refresh file file with:
# cp bazel-genfiles/third_party/tensorflow_decision_forests/tensorflow/ops/training/op.py third_party/tensorflow_decision_forests/tensorflow/ops/training/op.py
py_library(
    name = "op_py_precompile",
    srcs = ["op.py"],
    srcs_version = "PY3",
    deps = [
        "@org_tensorflow//tensorflow/python",
    ],
)

# Declaration of the ops.
cc_library(
    name = "op",
    srcs = ["op.cc"],
    linkstatic = 1,
    deps = ["@org_tensorflow//tensorflow/core:framework_headers_lib"],
    alwayslink = 1,
)

# Definition of the ops i.e. the kernels.
cc_library(
    name = "kernel",
    srcs = ["kernel.cc"],
    hdrs = [
        "features.h",
        "kernel.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core/platform:path",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_inference",
        "@ydf//yggdrasil_decision_forests/dataset:vertical_dataset",
        "@ydf//yggdrasil_decision_forests/learner:abstract_learner",
        "@ydf//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "@ydf//yggdrasil_decision_forests/learner:learner_library",
        "@ydf//yggdrasil_decision_forests/model:abstract_model",
        "@ydf//yggdrasil_decision_forests/model:model_library",
        "@ydf//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "@ydf//yggdrasil_decision_forests/utils:tensorflow",
    ],
)

# Tests
# =====
