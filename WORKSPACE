load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# ==========================================
#  Start of TensorFlow and its dependencies
# ==========================================

# Note: The OPs dynamic library depends on symbols specific to the version of
# absl used by tensorflow.
http_archive(
    name = "org_tensorflow",
    #sha256 = "b91ec194ddf6c4a5a2f9d1db4af4daab0b187ff691e6f88142413d2c7e77a3bb",
    #strip_prefix = "tensorflow-2.4.1",
    #urls = ["https://github.com/tensorflow/tensorflow/archive/v2.4.1.zip"],

    sha256 = "f7ad0a488559ee01f042a967c065482b074e11afd0299facbd8dc0cba9ae3fa9",
    strip_prefix = "tensorflow-2.5.0-rc3",
    urls = ["https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.5.0-rc3.zip"],

    #urls = ["https://github.com/tensorflow/tensorflow/archive/master.zip"],
    #strip_prefix = "tensorflow-master",
)

# Inject tensorflow dependencies.
# TensorFlow cannot anymore be injected from a sub-module.
# Note: The other is important.
load("@org_tensorflow//tensorflow:workspace3.bzl", tf1 = "workspace")
tf1()
load("@org_tensorflow//tensorflow:workspace2.bzl", tf2 = "workspace")
tf2()
load("@org_tensorflow//tensorflow:workspace1.bzl", tf3 = "workspace")
tf3()
load("@org_tensorflow//tensorflow:workspace0.bzl", tf4 = "workspace")
tf4()

# ========================================
#  End of TensorFlow and its dependencies
# ========================================

# Third party libraries
load("//third_party/absl_py:workspace.bzl", absl_py = "deps")
load("//third_party/absl:workspace.bzl", absl = "deps")
load("//third_party/gtest:workspace.bzl", gtest = "deps")
load("//third_party/protobuf:workspace.bzl", protobuf = "deps")

absl()
absl_py()
gtest()
protobuf()

# Yggdrasil Decision Forests
load("//third_party/yggdrasil_decision_forests:workspace.bzl", yggdrasil_decision_forests = "deps")
yggdrasil_decision_forests()

load("@ydf//yggdrasil_decision_forests:library.bzl", ydf_load_deps = "load_dependencies")
ydf_load_deps(
    exclude_repo = [
        "absl",
        "protobuf",
        "zlib",
    ],
    repo_name = "@ydf",
)
