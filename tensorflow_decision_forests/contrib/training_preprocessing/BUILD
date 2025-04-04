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
        "//tensorflow_decision_forests",
        "@pypi_absl_py//:pkg",  # /logging
        "@pypi_numpy//:pkg",
        "@release_or_nightly//:tensorflow_pkg",
    ],
)
