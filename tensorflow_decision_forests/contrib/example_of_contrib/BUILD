package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "example_of_contrib",
    srcs = [
        "__init__.py",
        "example_of_contrib.py",
    ],
    deps = [
        "//tensorflow_decision_forests",
    ],
)

# Tests
# =====

py_test(
    name = "example_of_contrib_test",
    srcs = ["example_of_contrib_test.py"],
    deps = [
        ":example_of_contrib",
        "@release_or_nightly//:tensorflow_pkg",
    ],
)
