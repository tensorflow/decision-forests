
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
    srcs_version = "PY3",
    deps = [
        "//tensorflow_decision_forests",
    ],
)

# Tests
# =====

py_test(
    name = "example_of_contrib_test",
    srcs = ["example_of_contrib_test.py"],
    python_version = "PY3",
    deps = [
        ":example_of_contrib",
        # TensorFlow Python,
    ],
)
