# Implementation of the Yggdrasil Distribute API using TensorFlow Distribution Strategies.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Worker binaries
# ===============

py_binary(
    name = "tensorflow_std_server_py",
    srcs = ["tensorflow_std_server.py"],
    main = "tensorflow_std_server.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # absl:app dep,
        # absl/flags dep,
        # absl/logging dep,
        "@release_or_nightly//:tensorflow_pkg",
        "//tensorflow_decision_forests",
    ],
)
