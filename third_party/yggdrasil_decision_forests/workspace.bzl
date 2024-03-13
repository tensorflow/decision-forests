"""Yggdrasil Decision Forests project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps(from_git_repo = True):
    if from_git_repo:
        YDF_VERSION = "1.9.0"
        YDF_SHA = "4b102dc3a08989aa069e1a830f58331e0b17d0a73df98bc3351e3378f2cfebc2"
        http_archive(
            name = "ydf",
            urls = ["https://github.com/google/yggdrasil-decision-forests/archive/refs/tags/v{version}.tar.gz".format(version = YDF_VERSION)],
            strip_prefix = "yggdrasil-decision-forests-{version}".format(version = YDF_VERSION),
            sha256 = YDF_SHA,
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
