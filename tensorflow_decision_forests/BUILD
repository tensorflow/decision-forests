
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

# If set, use the pre-compiled version of the OP wrappers. This significantly
# speed-up the compilation of the project.
config_setting(
    name = "use_precompiled_wrappers",
    values = {"define": "use_precompiled_wrappers=1"},
)

# If set, stop the training op when receiving an interrupt signal.
config_setting(
    name = "stop_training_on_interrupt",
    values = {"define": "stop_training_on_interrupt=1"},
)
