load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(["wrapper_main.cc"])

# Bzl Libraries
# =============

bzl_library(
    name = "wrapper_bzl",
    srcs = ["wrapper.bzl"],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_bzl",
        "//tensorflow_decision_forests/tensorflow:utils_bzl",
    ],
)

# Binaries
# ========

# Libraries
# =========

cc_library(
    name = "wrapper",
    srcs = ["wrapper.cc"],
    hdrs = ["wrapper.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:framework",
        "@ydf//yggdrasil_decision_forests/learner:learner_library",
        "@ydf//yggdrasil_decision_forests/utils:compatibility",
        "@ydf//yggdrasil_decision_forests/utils:status_macros",
    ],
)

# Tests
# =====

tf_cc_test(
    name = "wrapper_test",
    srcs = ["wrapper_test.cc"],
    linkstatic = True,
    deps = [
        ":wrapper",
        "@com_google_googletest//:gtest_main",
        "@ydf//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "@ydf//yggdrasil_decision_forests/learner/random_forest",
    ],
)
