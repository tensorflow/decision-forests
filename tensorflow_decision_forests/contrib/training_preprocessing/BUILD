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
        # TensorFlow Python,
        "//tensorflow_decision_forests",
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
        # numpy dep,
        # TensorFlow Python,
        "//tensorflow_decision_forests",
    ],
)
