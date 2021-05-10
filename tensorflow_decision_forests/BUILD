
package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(["LICENSE"])

py_library(
    name = "tensorflow_decision_forests",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_decision_forests/component/builder",
        "//tensorflow_decision_forests/component/inspector",
        "//tensorflow_decision_forests/component/model_plotter",
        "//tensorflow_decision_forests/component/py_tree",
        "//tensorflow_decision_forests/keras",
    ],
)
