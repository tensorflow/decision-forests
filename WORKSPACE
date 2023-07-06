load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# ==========================================
#  Start of TensorFlow and its dependencies
# ==========================================

# This version of TensorFlow is injected only to make sure we use the same dependencies as TensorFlow (protobuffer, grpc, absl).
# TensorFlow is not compiled.

# Note: The OPs dynamic library depends on symbols specific to the version of
# absl used by tensorflow.
http_archive(
    name = "org_tensorflow",
    strip_prefix = "tensorflow-2.13.0",
    sha256 = "447cdb65c80c86d6c6cf1388684f157612392723eaea832e6392d219098b49de",
    urls = ["https://github.com/tensorflow/tensorflow/archive/v2.13.0.zip"],
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

# Inject TensorFlow from the Pypi package.
load("//third_party/tensorflow_pypi:tf_configure.bzl", "tf_configure")

tf_configure(name = "tensorflow_pypi")

# ========================================
#  End of TensorFlow and its dependencies
# ========================================

# Third party libraries
load("//third_party/absl_py:workspace.bzl", absl_py = "deps")
load("//third_party/absl:workspace.bzl", absl = "deps")
load("//third_party/benchmark:workspace.bzl", benchmark = "deps")
load("//third_party/gtest:workspace.bzl", gtest = "deps")
load("//third_party/protobuf:workspace.bzl", protobuf = "deps")

absl()
absl_py()
benchmark()
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
        "farmhash",
        "grpc",
    ],
    repo_name = "@ydf",
)
