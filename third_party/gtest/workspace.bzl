"""Google Test project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    http_archive(
        name = "com_google_googletest",
        urls = ["https://github.com/google/googletest/archive/011959aafddcd30611003de96cfd8d7a7685c700.zip"],
        strip_prefix = "googletest-011959aafddcd30611003de96cfd8d7a7685c700",
        sha256 = "6a5d7d63cd6e0ad2a7130471105a3b83799a7a2b14ef7ec8d742b54f01a4833c",
    )
