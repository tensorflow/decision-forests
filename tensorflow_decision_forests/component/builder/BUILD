package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "builder",
    srcs = ["builder.py"],
    deps = [
        "//tensorflow_decision_forests/component/inspector",
        "//tensorflow_decision_forests/component/inspector:blob_sequence",
        "//tensorflow_decision_forests/component/py_tree",
        "//tensorflow_decision_forests/keras:core_inference",
        "//tensorflow_decision_forests/tensorflow:core_inference",
        "@pypi_numpy//:pkg",
        "@pypi_six//:pkg",
        "@release_or_nightly//:tensorflow_pkg",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf//yggdrasil_decision_forests/model:abstract_model_py_proto",
        "@ydf//yggdrasil_decision_forests/model/gradient_boosted_trees:gradient_boosted_trees_py_proto",
        "@ydf//yggdrasil_decision_forests/model/random_forest:random_forest_py_proto",
    ],
)

# Tests
# =====

py_test(
    name = "builder_test",
    timeout = "long",
    srcs = ["builder_test.py"],
    data = ["@ydf//yggdrasil_decision_forests/test_data"],
    shard_count = 10,
    deps = [
        ":builder",
        "//tensorflow_decision_forests/component/inspector",
        "//tensorflow_decision_forests/component/py_tree",
        "//tensorflow_decision_forests/keras",
        "@pypi_absl_py//:pkg",  # /flags
        "@pypi_numpy//:pkg",
        "@pypi_pandas//:pkg",
        "@release_or_nightly//:tensorflow_pkg",
    ],
)
