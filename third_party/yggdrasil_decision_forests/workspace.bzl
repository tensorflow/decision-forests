"""Yggdrasil Decision Forests project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps(from_git_repo = True):
    if from_git_repo:
        http_archive(
            name = "ydf",
            urls = ["https://github.com/google/yggdrasil-decision-forests/archive/refs/heads/main.zip"],
            strip_prefix = "yggdrasil-decision-forests-main",
        )
    else:
        # You can also clone the YDF repository manually.
        native.local_repository(
            name = "ydf",
            path = "../yggdrasil-decision-forests",
        )
