package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_binary(
    name = "minimal",
    srcs = ["minimal.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # absl:app dep,
        "@pypi_numpy//:pkg",
        "@pypi_pandas//:pkg",
        "@release_or_nightly//:tensorflow_pkg",
        "//tensorflow_decision_forests",
        "@release_or_nightly//:tf_keras_pkg",
    ],
)

py_binary(
    name = "hyperparameter_optimization",
    srcs = ["hyperparameter_optimization.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # absl:app dep,
        "@pypi_numpy//:pkg",
        "@pypi_pandas//:pkg",
        "@release_or_nightly//:tensorflow_pkg",
        "//tensorflow_decision_forests",
        "@release_or_nightly//:tf_keras_pkg",
    ],
)

py_binary(
    name = "distributed_training",
    srcs = ["distributed_training.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # absl:app dep,
        # absl/logging dep,
        "@pypi_numpy//:pkg",
        "@pypi_pandas//:pkg",
        "@release_or_nightly//:tensorflow_pkg",
        "//tensorflow_decision_forests",
        "@release_or_nightly//:tf_keras_pkg",
    ],
)

py_binary(
    name = "distributed_hyperparameter_optimization",
    srcs = ["distributed_hyperparameter_optimization.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # absl:app dep,
        # absl/logging dep,
        "@release_or_nightly//:tensorflow_pkg",
        "//tensorflow_decision_forests",
        "@release_or_nightly//:tf_keras_pkg",
    ],
)
