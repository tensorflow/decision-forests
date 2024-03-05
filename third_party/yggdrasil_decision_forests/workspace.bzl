"""Yggdrasil Decision Forests project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps(from_git_repo = True):
    if from_git_repo:
        http_archive(
            name = "ydf",
            urls = ["https://github.com/google/yggdrasil-decision-forests/archive/refs/heads/main.zip"],
            strip_prefix = "yggdrasil-decision-forests-main",
            # patch_args = ["-p1"],
            # patches = ["@ydf//yggdrasil_decision_forests:ydf.patch"],
        )
    else:
        # You can also clone the YDF repository manually.
        # Note that you need to manually apply the patch for Tensorflow >= 2.16 or nightly.
        native.local_repository(
            name = "ydf",
            # When downloading from Github, you might need - instead of _ as folder name
            path = "../yggdrasil_decision_forests",
        )
