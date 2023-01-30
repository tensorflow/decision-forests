load("//tools/build_defs/license:license.bzl", "license")

package(
    default_applicable_licenses = ["//tensorflow_decision_forests:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

license(
    name = "license",
    package_name = "tensorflow_decision_forests",
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
        "//tensorflow_decision_forests/component/tuner",
        "//tensorflow_decision_forests/keras",
        "//tensorflow_decision_forests/tensorflow:check_version",
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

# If "disable_tf_ps_distribution_strategy" is true, the TF Parameter Server
# distribution strategy is not available for distributed training.
#
# Distribution with TF PS is currently NOT supported for OSS TF-DF with shared
# build (monolithic build works however) and TF<2.7. In this case, the GRPC
# Worker Server can be used instead.
config_setting(
    name = "disable_tf_ps_distribution_strategy",
    values = {"define": "tf_ps_distribution_strategy=0"},
)
