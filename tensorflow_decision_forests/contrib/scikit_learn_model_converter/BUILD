package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "scikit_learn_model_converter",
    srcs = [
        "__init__.py",
        "scikit_learn_model_converter.py",
    ],
    deps = [
        "//tensorflow_decision_forests",
        "@pypi_scikit_learn//:pkg",
        "@release_or_nightly//:tensorflow_pkg",
    ],
)

# Tests
# =====

py_test(
    name = "scikit_learn_model_converter_test",
    srcs = ["scikit_learn_model_converter_test.py"],
    deps = [
        ":scikit_learn_model_converter",
        "@pypi_absl_py//:pkg",  # /testing:parameterized
        "@pypi_numpy//:pkg",
        "@pypi_scikit_learn//:pkg",
        "@release_or_nightly//:tensorflow_pkg",
    ],
)
