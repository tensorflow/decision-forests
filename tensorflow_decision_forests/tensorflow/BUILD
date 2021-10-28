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
        "@ydf//yggdrasil_decision_forests/model/random_forest",
        "@ydf//yggdrasil_decision_forests/serving/decision_forest:register_engines",
    ],
    alwayslink = 1,
)

cc_library(
    name = "canonical_learners",
    deps = [
        "@ydf//yggdrasil_decision_forests/learner/cart",
        "@ydf//yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees",
        "@ydf//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "@ydf//yggdrasil_decision_forests/learner/random_forest",
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
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # pandas dep,
        "@org_tensorflow//tensorflow/python",
        "@org_tensorflow//tensorflow/python/distribute:distribute_lib",
        "@org_tensorflow//tensorflow/python/distribute:parameter_server_strategy_v2",
        "@org_tensorflow//tensorflow/python/distribute/coordinator:cluster_coordinator",
        "@org_tensorflow//tensorflow/python/distribute/coordinator:coordinator_context",
        "//tensorflow_decision_forests/tensorflow/distribute:tf_distribution_py_proto",
        "//tensorflow_decision_forests/tensorflow/ops/training:api_py",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "@ydf//yggdrasil_decision_forests/model:abstract_model_py_proto",
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
        # pandas dep,
        "@org_tensorflow//tensorflow/python",
        "@ydf//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
    ],
)
