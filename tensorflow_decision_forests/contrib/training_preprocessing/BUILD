package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "training_preprocessing",
    srcs = [
        "__init__.py",
        "training_preprocessing.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_decision_forests",
        "@release_or_nightly//:tensorflow_pkg",
    ],
)

# Tests
# =====

py_test(
    name = "training_preprocessing_test",
    srcs = ["training_preprocessing_test.py"],
    deps = [
        ":training_preprocessing",
        # absl/logging dep,
        # absl/testing:parameterized dep,
        "@pypi_numpy//:pkg",
        "@release_or_nightly//:tensorflow_pkg",
        "//tensorflow_decision_forests",
    ],
)
