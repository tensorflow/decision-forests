package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "tuner",
    srcs = ["tuner.py"],
    deps = [
        "@ydf//yggdrasil_decision_forests/learner/hyperparameters_optimizer:hyperparameters_optimizer_py_proto",
        "@ydf//yggdrasil_decision_forests/learner/hyperparameters_optimizer/optimizers:random_py_proto",
        "@ydf//yggdrasil_decision_forests/model:hyperparameter_py_proto",
    ],
)

# Tests
# =====

py_test(
    name = "tuner_test",
    timeout = "long",
    srcs = ["tuner_test.py"],
    data = ["@ydf//yggdrasil_decision_forests/test_data"],
    shard_count = 4,
    deps = [
        ":tuner",
        "@com_google_protobuf//:python_srcs",
        # absl/flags dep,
        # absl/logging dep,
        # absl/testing:parameterized dep,
        "@pypi_pandas//:pkg",
        "@release_or_nightly//:tensorflow_pkg",
        "//tensorflow_decision_forests/keras",
        "@ydf//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
    ],
)
