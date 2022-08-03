"""Rule generation utilities."""

load("@org_tensorflow//tensorflow:tensorflow.bzl", "if_not_windows", "tf_binary_additional_srcs", "tf_cc_binary", "tf_copts")
load("//tensorflow_decision_forests/tensorflow:utils.bzl", "rpath_linkopts_to_tensorflow")

def py_wrap_yggdrasil_learners(
        name = None,
        learner_deps = []):
    """Creates Keras wrappers around Yggdrasil Decision Forest (YDF) learners.

    Creates a py_library called "{name}" and containing the file "{name}.py".
    This library introduces a TensorFlow Decision Forests (TFDF) Keras class
    wrapping for each YDF learner defined in "learner_deps". The constructor of
    these classes contains a argument for the learner generic hyper-parameter.

    For example, if "learner_deps" contains a c++ dependency that register a
    learner with a key equal to "RANDOM_FOREST", the wrapper will create a
    python class called "RandomForestModel" deriving the base TFDF model class.

    Args:
        name: Name of the rule.
        learner_deps: List of dependencies linking Yggdrasil Decision Forest
          learners.
    """

    # Absolute path to the wrapper generator directory.
    wrapper_package = "//tensorflow_decision_forests/keras/wrapper"

    # Filename of the wrapper generator source code in the user package.
    local_cc_main = name + "_wrapper_main.cc"

    # Target name of the wrapper generator binary.
    wrapper_name = name + "_wrapper_main"

    # Target name of the command running the wrapper generator.
    run_wrapper_name = name + "_run_wrapper"

    # Copy the wrapper main source code to the user package.
    native.genrule(
        name = name + "_copy_cc_main",
        outs = [local_cc_main],
        srcs = [wrapper_package + ":wrapper_main.cc"],
        cmd = "cp $< $@",
    )

    # Compiles the wrapper binary.
    # TODO: Find way to link from pypi.
    # Note: This rule will compile a small part of TF.
    tf_cc_binary(
        name = wrapper_name,
        copts = tf_copts(),
        linkopts = if_not_windows(["-lm", "-Wl,-ldl"]) + rpath_linkopts_to_tensorflow(wrapper_name),
        srcs = [":" + local_cc_main],
        deps = [
            wrapper_package + ":wrapper",
        ] + learner_deps,
        linkstatic = 1,
    )

    # Runs the wrapper binary and generate the wrapper .py source code.
    native.genrule(
        name = run_wrapper_name,
        srcs = [],
        outs = [name + ".py"],
        cmd = "$(location " + wrapper_name + ") > \"$@\"",
        tools = [":" + wrapper_name] + tf_binary_additional_srcs(),
    )

    # Python library around the generated .py source code.
    native.py_library(
        name = name,
        srcs = [name + ".py"],
        srcs_version = "PY3",
        deps = [
            "//tensorflow_decision_forests/keras:core",
            "//tensorflow_decision_forests/component/tuner",
            # TensorFlow Python,
            "@ydf//yggdrasil_decision_forests/model:abstract_model_py_proto",
            "@ydf//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        ],
        data = [":" + run_wrapper_name, ":" + wrapper_name],
    )
