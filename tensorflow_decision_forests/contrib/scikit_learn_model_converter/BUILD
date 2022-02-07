
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
    srcs_version = "PY3",
    deps = [
        "//third_party/py/sklearn",
        # TensorFlow Python,
        "//third_party/py/tensorflow_decision_forests",
    ],
)

# Tests
# =====

py_test(
    name = "scikit_learn_model_converter_test",
    srcs = ["scikit_learn_model_converter_test.py"],
    srcs_version = "PY3",
    deps = [
        ":scikit_learn_model_converter",
        "//testing/pybase:parameterized",
        # numpy dep,
        "//third_party/py/sklearn",
        # TensorFlow Python,
    ],
)
