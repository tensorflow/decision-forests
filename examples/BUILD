package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_binary(
    name = "minimal",
    srcs = ["minimal.py"],
    deps = [
        "//tensorflow_decision_forests",
        "@pypi_absl_py//:pkg",  # :app
        "@pypi_numpy//:pkg",
        "@pypi_pandas//:pkg",
        "@release_or_nightly//:tensorflow_pkg",
        "@release_or_nightly//:tf_keras_pkg",
    ],
)

py_binary(
    name = "hyperparameter_optimization",
    srcs = ["hyperparameter_optimization.py"],
    deps = [
        "//tensorflow_decision_forests",
        "@pypi_absl_py//:pkg",  # :app
        "@pypi_numpy//:pkg",
        "@pypi_pandas//:pkg",
        "@release_or_nightly//:tensorflow_pkg",
        "@release_or_nightly//:tf_keras_pkg",
    ],
)

py_binary(
    name = "distributed_training",
    srcs = ["distributed_training.py"],
    deps = [
        "//tensorflow_decision_forests",
        "@pypi_absl_py//:pkg",  # :app
        "@pypi_numpy//:pkg",
        "@pypi_pandas//:pkg",
        "@release_or_nightly//:tensorflow_pkg",
        "@release_or_nightly//:tf_keras_pkg",
    ],
)

py_binary(
    name = "distributed_hyperparameter_optimization",
    srcs = ["distributed_hyperparameter_optimization.py"],
    deps = [
        "//tensorflow_decision_forests",
        "@pypi_absl_py//:pkg",  # :app
        "@release_or_nightly//:tensorflow_pkg",
        "@release_or_nightly//:tf_keras_pkg",
    ],
)
