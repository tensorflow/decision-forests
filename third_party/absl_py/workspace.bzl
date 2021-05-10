"""Absl project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    http_archive(
        name = "com_google_absl_py",
        urls = ["https://github.com/abseil/abseil-py/archive/master.zip"],
        strip_prefix = "abseil-py-master",
    )
