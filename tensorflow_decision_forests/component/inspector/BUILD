package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "inspector",
    srcs = ["inspector.py"],
    deps = [
        ":blob_sequence",
        "//tensorflow_decision_forests/component/py_tree",
        "@pypi_pandas//:pkg",
        "@pypi_six//:pkg",
        "@release_or_nightly//:tensorflow_pkg",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf//yggdrasil_decision_forests/metric:metric_py_proto",
        "@ydf//yggdrasil_decision_forests/model:abstract_model_py_proto",
        "@ydf//yggdrasil_decision_forests/model:hyperparameter_py_proto",
        "@ydf//yggdrasil_decision_forests/model/decision_tree:decision_tree_py_proto",
        "@ydf//yggdrasil_decision_forests/model/gradient_boosted_trees:gradient_boosted_trees_py_proto",
        "@ydf//yggdrasil_decision_forests/model/multitasker:multitasker_py_proto",
        "@ydf//yggdrasil_decision_forests/model/random_forest:random_forest_py_proto",
    ],
)

py_library(
    name = "blob_sequence",
    srcs = ["blob_sequence.py"],
    deps = [
        "@release_or_nightly//:tensorflow_pkg",
    ],
)

# Tests
# =====

py_test(
    name = "inspector_test",
    timeout = "long",
    srcs = ["inspector_test.py"],
    data = ["@ydf//yggdrasil_decision_forests/test_data"],
    shard_count = 10,
    deps = [
        ":inspector",
        "//tensorflow_decision_forests/component/py_tree",
        "//tensorflow_decision_forests/keras",
        "@pypi_absl_py//:pkg",  # /flags
        "@pypi_numpy//:pkg",
        "@release_or_nightly//:tensorflow_pkg",
        "@ydf//yggdrasil_decision_forests/metric:metric_py_proto",
        "@ydf//yggdrasil_decision_forests/model/gradient_boosted_trees:gradient_boosted_trees_py_proto",
    ],
)

py_test(
    name = "blob_sequence_test",
    srcs = ["blob_sequence_test.py"],
    data = ["@ydf//yggdrasil_decision_forests/test_data"],
    deps = [
        ":blob_sequence",
        "@pypi_absl_py//:pkg",  # /flags
        "@release_or_nightly//:tensorflow_pkg",
        "@ydf//yggdrasil_decision_forests/model/decision_tree:decision_tree_py_proto",
    ],
)
