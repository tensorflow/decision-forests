workspace(name = "org_tensorflow_decision_forests")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# rules_java, rules_shell are required for Tensorflow.
http_archive(
    name = "rules_java",
    sha256 = "c73336802d0b4882e40770666ad055212df4ea62cfa6edf9cb0f9d29828a0934",
    url = "https://github.com/bazelbuild/rules_java/releases/download/5.3.5/rules_java-5.3.5.tar.gz",
)

http_archive(
    name = "rules_shell",
    sha256 = "bc61ef94facc78e20a645726f64756e5e285a045037c7a61f65af2941f4c25e1",
    strip_prefix = "rules_shell-0.4.1",
    url = "https://github.com/bazelbuild/rules_shell/releases/download/v0.4.1/rules_shell-v0.4.1.tar.gz",
)

# ==========================================
#  Start of TensorFlow and its dependencies
# ==========================================

# This version of TensorFlow is injected only to make sure we use the same dependencies as TensorFlow (protobuffer, grpc, absl).
# TensorFlow is not compiled.

# Note: The OPs dynamic library depends on symbols specific to the version of
# absl used by tensorflow.
http_archive(
    name = "org_tensorflow",
    sha256 = "4691b18e8c914cdf6759b80f1b3b7f3e17be41099607ed0143134f38836d058e",
    strip_prefix = "tensorflow-2.19.0",
    urls = ["https://github.com/tensorflow/tensorflow/archive/v2.19.0.tar.gz"],
)

load("//tensorflow_decision_forests:tensorflow_decision_forests.bzl", "py_deps_profile")

py_deps_profile(
    name = "release_or_nightly",
    deps_map = {
        "tensorflow": [
            "tf-nightly",
            "tf_header_lib",
            "libtensorflow_framework",
        ],
        "tf-keras": ["tf-keras-nightly"],
    },
    pip_repo_name = "pypi",
    requirements_in = "//configure:requirements.in",
    switch = {
        "IS_NIGHTLY": "nightly",
    },
)

# Initialize ML Toolchains
# Details: https://github.com/google-ml-infra/rules_ml_toolchain
http_archive(
    name = "rules_ml_toolchain",
    sha256 = "9dbee8f24cc1b430bf9c2a6661ab70cbca89979322ddc7742305a05ff637ab6b",
    strip_prefix = "rules_ml_toolchain-545c80f1026d526ea9c7aaa410bf0b52c9a82e74",
    urls = [
        "https://github.com/google-ml-infra/rules_ml_toolchain/archive/545c80f1026d526ea9c7aaa410bf0b52c9a82e74.tar.gz",
    ],
)

load(
    "@rules_ml_toolchain//cc/deps:cc_toolchain_deps.bzl",
    "cc_toolchain_deps",
)

cc_toolchain_deps()

# Initialize hermetic Python
load("@org_tensorflow//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@org_tensorflow//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    default_python_version = "system",
    requirements = {
        "3.9": "//configure:requirements_lock_3_9.txt",
        "3.10": "//configure:requirements_lock_3_10.txt",
        "3.11": "//configure:requirements_lock_3_11.txt",
    },
)

load("@org_tensorflow//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("//third_party/tensorflow_pypi:tf_configure.bzl", "tf_configure")

tf_configure()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

# Inject tensorflow dependencies.
# TensorFlow cannot anymore be injected from a sub-module.
# Note: The order is important.
load("@org_tensorflow//tensorflow:workspace3.bzl", tf1 = "workspace")

tf1()

load("@org_tensorflow//tensorflow:workspace2.bzl", tf2 = "workspace")

tf2()

load("@org_tensorflow//tensorflow:workspace1.bzl", tf3 = "workspace")

tf3()

load("@org_tensorflow//tensorflow:workspace0.bzl", tf4 = "workspace")

tf4()

load(
    "@local_xla//third_party/py:python_wheel.bzl",
    "python_wheel_version_suffix_repository",
)

python_wheel_version_suffix_repository(name = "tf_wheel_version_suffix")

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//gpu/cuda:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

# ========================================
#  End of TensorFlow and its dependencies
# ========================================

# Third party libraries
load("//third_party/absl:workspace.bzl", absl = "deps")
load("//third_party/absl_py:workspace.bzl", absl_py = "deps")
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
        "eigen",
        "pybind11",
        "pybind11_abseil",
        "pybind11_protobuf",
        "tensorflow",
    ],
    repo_name = "@ydf",
)
