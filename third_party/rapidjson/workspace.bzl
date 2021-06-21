"""rapidjson project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps(prefix = ""):
    http_archive(
        name = "com_github_tencent_rapidjson",
        url = "https://github.com/Tencent/rapidjson/archive/v1.1.0.zip",
        sha256 = "8e00c38829d6785a2dfb951bb87c6974fa07dfe488aa5b25deec4b8bc0f6a3ab",
        strip_prefix = "rapidjson-1.1.0",
        build_file = prefix + "//third_party/rapidjson:rapidjson.BUILD",
    )
