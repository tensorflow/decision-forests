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
    deps = [
        "//tensorflow_decision_forests/component/inspector",
        "//tensorflow_decision_forests/component/py_tree:condition",
        "//tensorflow_decision_forests/component/py_tree:node",
        "//tensorflow_decision_forests/component/py_tree:tree",
        "//tensorflow_decision_forests/component/py_tree:value",
        "@release_or_nightly//:tensorflow_pkg",
    ],
)

# Tests
# =====

py_test(
    name = "model_plotter_test",
    srcs = ["model_plotter_test.py"],
    deps = [
        ":model_plotter",
        "//tensorflow_decision_forests/component/py_tree:condition",
        "//tensorflow_decision_forests/component/py_tree:dataspec",
        "//tensorflow_decision_forests/component/py_tree:node",
        "//tensorflow_decision_forests/component/py_tree:tree",
        "//tensorflow_decision_forests/component/py_tree:value",
        "@pypi_absl_py//:pkg",  # /flags
        "@release_or_nightly//:tensorflow_pkg",
    ],
)
