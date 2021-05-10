
package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "builder",
    srcs = ["builder.py"],
    srcs_version = "PY3",
    deps = [
        # six dep,
        "@org_tensorflow//tensorflow/python",
        "//tensorflow_decision_forests/component/inspector",
        "//tensorflow_decision_forests/component/inspector:blob_sequence",
        "//tensorflow_decision_forests/component/py_tree",
        "//tensorflow_decision_forests/keras:core",
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
    python_version = "PY3",
    shard_count = 10,
    deps = [
        ":builder",
        # absl/flags dep,
        # absl/logging dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        # pandas dep,
        "@org_tensorflow//tensorflow/python",
        "//tensorflow_decision_forests/component/inspector",
        "//tensorflow_decision_forests/component/py_tree",
        "//tensorflow_decision_forests/keras",
    ],
)
