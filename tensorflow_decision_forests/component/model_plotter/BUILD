
package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Data
# ====

filegroup(
    name = "assets",
    srcs = ["plotter.js"],
)

# Libraries
# =========

py_library(
    name = "model_plotter",
    srcs = ["model_plotter.py"],
    data = ["assets"],
    srcs_version = "PY3",
    deps = [
        "@org_tensorflow//tensorflow/python",
        "//tensorflow_decision_forests/component/inspector",
        "//tensorflow_decision_forests/component/py_tree:condition",
        "//tensorflow_decision_forests/component/py_tree:node",
        "//tensorflow_decision_forests/component/py_tree:tree",
        "//tensorflow_decision_forests/component/py_tree:value",
        "//tensorflow_decision_forests/keras:core",
    ],
)

# Tests
# =====

py_test(
    name = "model_plotter_test",
    srcs = ["model_plotter_test.py"],
    python_version = "PY3",
    deps = [
        ":model_plotter",
        # absl/flags dep,
        # absl/logging dep,
        # absl/testing:parameterized dep,
        "@org_tensorflow//tensorflow/python",
        "//tensorflow_decision_forests/component/py_tree:condition",
        "//tensorflow_decision_forests/component/py_tree:dataspec",
        "//tensorflow_decision_forests/component/py_tree:node",
        "//tensorflow_decision_forests/component/py_tree:tree",
        "//tensorflow_decision_forests/component/py_tree:value",
    ],
)
