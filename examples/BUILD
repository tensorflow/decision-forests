
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
        # numpy dep,
        # pandas dep,
        # TensorFlow Python,
        "//tensorflow_decision_forests",
    ],
)

py_binary(
    name = "hyperparameter_optimization",
    srcs = ["hyperparameter_optimization.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # absl:app dep,
        # numpy dep,
        # pandas dep,
        # TensorFlow Python,
        "//tensorflow_decision_forests",
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
        # numpy dep,
        # pandas dep,
        # TensorFlow Python,
        "//tensorflow_decision_forests",
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
        # TensorFlow Python,
        "//tensorflow_decision_forests",
    ],
)
