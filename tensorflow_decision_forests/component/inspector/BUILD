
package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "inspector",
    srcs = ["inspector.py"],
    srcs_version = "PY3",
    deps = [
        ":blob_sequence",
        # six dep,
        # TensorFlow Python,
        "//tensorflow_decision_forests/component/py_tree",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf//yggdrasil_decision_forests/metric:metric_py_proto",
        "@ydf//yggdrasil_decision_forests/model:abstract_model_py_proto",
        "@ydf//yggdrasil_decision_forests/model/decision_tree:decision_tree_py_proto",
        "@ydf//yggdrasil_decision_forests/model/gradient_boosted_trees:gradient_boosted_trees_py_proto",
        "@ydf//yggdrasil_decision_forests/model/random_forest:random_forest_py_proto",
    ],
)

py_library(
    name = "blob_sequence",
    srcs = ["blob_sequence.py"],
    srcs_version = "PY3",
    deps = [
        # TensorFlow Python,
    ],
)

# Tests
# =====

py_test(
    name = "inspector_test",
    timeout = "long",
    srcs = ["inspector_test.py"],
    data = ["@ydf//yggdrasil_decision_forests/test_data"],
    python_version = "PY3",
    shard_count = 10,
    deps = [
        ":inspector",
        # absl/flags dep,
        # absl/logging dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        # pandas dep,
        # TensorFlow Python,
        "//tensorflow_decision_forests/component/py_tree",
        "//tensorflow_decision_forests/keras",
        "@ydf//yggdrasil_decision_forests/metric:metric_py_proto",
        "@ydf//yggdrasil_decision_forests/model/gradient_boosted_trees:gradient_boosted_trees_py_proto",
    ],
)

py_test(
    name = "blob_sequence_test",
    srcs = ["blob_sequence_test.py"],
    data = ["@ydf//yggdrasil_decision_forests/test_data"],
    python_version = "PY3",
    deps = [
        ":blob_sequence",
        # absl/flags dep,
        # absl/logging dep,
        # absl/testing:parameterized dep,
        # TensorFlow Python,
        "@ydf//yggdrasil_decision_forests/model/decision_tree:decision_tree_py_proto",
    ],
)
