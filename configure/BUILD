load("@python//:defs.bzl", "compile_pip_requirements")
load("@python_version_repo//:py_version.bzl", "REQUIREMENTS")

licenses(["notice"])

compile_pip_requirements(
    name = "requirements",
    extra_args = [
        "--allow-unsafe",
        "--build-isolation",
        "--rebuild",
    ],
    generate_hashes = True,
    requirements_in = "//configure:requirements.in",
    requirements_txt = REQUIREMENTS,
)

py_binary(
    name = "tensorflow_build_info",
    srcs = ["tensorflow_build_info.py"],
    deps = ["@pypi_tf_nightly//:pkg"],
)
