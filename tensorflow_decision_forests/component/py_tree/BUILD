load("@rules_python//python:defs.bzl", "py_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# All the components
# =========

py_library(
    name = "py_tree",
    srcs = ["__init__.py"],
    deps = [
        ":condition",
        ":dataspec",
        ":node",
        ":objective",
        ":tree",
        ":value",
    ],
)

# Libraries
# =========

py_library(
    name = "tree",
    srcs = ["tree.py"],
    deps = [":node"],
)

py_library(
    name = "node",
    srcs = ["node.py"],
    deps = [
        ":condition",
        ":value",
        # absl/logging dep,
        "@pypi_numpy//:pkg",
        "@pypi_six//:pkg",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf//yggdrasil_decision_forests/model:abstract_model_py_proto",
        "@ydf//yggdrasil_decision_forests/model/decision_tree:decision_tree_py_proto",
    ],
)

py_library(
    name = "condition",
    srcs = ["condition.py"],
    deps = [
        ":dataspec",
        # absl/logging dep,
        "@pypi_numpy//:pkg",
        "@pypi_six//:pkg",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf//yggdrasil_decision_forests/model/decision_tree:decision_tree_py_proto",
    ],
)

py_library(
    name = "value",
    srcs = ["value.py"],
    deps = [
        "@pypi_numpy//:pkg",
        "@pypi_six//:pkg",
        "@ydf//yggdrasil_decision_forests/model/decision_tree:decision_tree_py_proto",
    ],
)

py_library(
    name = "dataspec",
    srcs = ["dataspec.py"],
    deps = [
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_py_proto",
    ],
)

py_library(
    name = "objective",
    srcs = ["objective.py"],
    deps = [
        "@pypi_six//:pkg",
        "@ydf//yggdrasil_decision_forests/model:abstract_model_py_proto",
    ],
)

# Tests
# =====

py_test(
    name = "tree_test",
    srcs = ["tree_test.py"],
    deps = [
        ":condition",
        ":dataspec",
        ":node",
        ":tree",
        ":value",
        # absl/logging dep,
        # absl/testing:parameterized dep,
        "@release_or_nightly//:tensorflow_pkg",
    ],
)

py_test(
    name = "node_test",
    srcs = ["node_test.py"],
    deps = [
        ":condition",
        ":dataspec",
        ":node",
        ":value",
        # absl/logging dep,
        # absl/testing:parameterized dep,
        "@release_or_nightly//:tensorflow_pkg",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf//yggdrasil_decision_forests/model/decision_tree:decision_tree_py_proto",
    ],
)

py_test(
    name = "condition_test",
    srcs = ["condition_test.py"],
    deps = [
        ":condition",
        ":dataspec",
        # absl/logging dep,
        # absl/testing:parameterized dep,
        "@release_or_nightly//:tensorflow_pkg",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf//yggdrasil_decision_forests/model/decision_tree:decision_tree_py_proto",
    ],
)

py_test(
    name = "dataspec_test",
    srcs = ["dataspec_test.py"],
    deps = [
        ":dataspec",
        # absl/testing:parameterized dep,
        "@release_or_nightly//:tensorflow_pkg",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_py_proto",
    ],
)

py_test(
    name = "value_test",
    srcs = ["value_test.py"],
    deps = [
        ":value",
        # absl/logging dep,
        # absl/testing:parameterized dep,
        "@release_or_nightly//:tensorflow_pkg",
        "@ydf//yggdrasil_decision_forests/model/decision_tree:decision_tree_py_proto",
    ],
)

py_test(
    name = "objective_test",
    srcs = ["objective_test.py"],
    deps = [
        ":dataspec",
        ":objective",
        # absl/logging dep,
        # absl/testing:parameterized dep,
        "@release_or_nightly//:tensorflow_pkg",
    ],
)
