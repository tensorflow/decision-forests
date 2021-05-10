"""Yggdrasil Decision Forests project."""

# load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    # http_archive(
    #      name = "yggdrasil_decision_forests",
    #      urls = ["https://github.com/google/yggdrasil-decision-forests/archive/master.zip"],
    #      strip_prefix = "yggdrasil_decision_forests-master",
    #  )

    # Assume a copy of YDF next to this workspace.
    native.local_repository(
        name = "ydf",
        path = "../yggdrasil_decision_forests_bazel",
    )
