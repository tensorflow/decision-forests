# Implementation of the Yggdrasil Distribute API using TensorFlow Distribution Strategies.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

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
    deps = [
        "//tensorflow_decision_forests",
        "@pypi_absl_py//:pkg",  # :app
        "@release_or_nightly//:tensorflow_pkg",
    ],
)
