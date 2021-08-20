# Experimental. Not ready for use.

This directory contains an implementation of a "Yggdrasil Decision Forests
Distribute" (YDF-D) manager using TensorFlow distributed computation.

Each YDF-D worker runs on a separate TensorFlow Server. This implementation does
not start the TF Servers itself, instead it assumes there are servers that have
been started by the TensorFlow distributed training system.

The address of each worker server is configured in one of two ways:

-   [TF_CONFIG](https://www.tensorflow.org/guide/distributed_training#setting_up_the_tf_config_environment_variable)
    environment variable, set in each worker before they are started: this will
    be set up with a JSON serialized
    `yaggdrasil_decision_forests.distribute.proto.Config` proto, with the
    `tf_distribution` extension.
-   A list of socket addresses.

See https://www.tensorflow.org/guide/distributed_training

TF-DF provides pre-configured server binaries in `:tf_distribution_py_worker`
with the required TF-DF custom ops. If using your own TF server binary, make
sure to link the custom c++ ops defined in `:register_ops`.
