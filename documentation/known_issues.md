# Known Issues

The underlying engine behind the decision forests algorithms used by TensorFlow
Decision Forests have been extensively production-tested. But this API to
TensorFlow and Keras is new, and some issues are expected -- we are trying to
fix them as quickly as possible.

See also the
[known issues of Yggdrasil Decision Forests](https://github.com/google/yggdrasil-decision-forests/documentation/known_issues.md)
and the [migration guide](migration.md) for behavior that is different from
other algorithms.

## Windows Pip package is not available

TensorFlow Decision Forest is not yet available as a Windows Pip package.

**Workarounds:**

-   *Solution #1:* Install
    [Windows Subsystem for Linux (WSL)](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux)
    on your Windows machine and follow the Linux instructions.

## No support for in-model preprocessing with input of rank 1

Keras expands the dimension of input tensors to rank 2 (using `tf.expand_dims`).
If your `preprocessing` model argument only support rank 1 tensors, you will get
an error complaining about tensor shape.

**Workarounds:**

-   *Solution #1:* Apply your preprocessing before the model, for example using
    the dataset's
    [`map`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map)
    method.

-   *Solution #2:* Wrapps your preprocessing function into another function that
    [squeeze](https://www.tensorflow.org/api_docs/python/tf/squeeze) its inputs.

## No support for TF distribution strategies.

TF-DF does not yet support distribution strategies or datasets that do not fit
in memory. This is because the classical decision forest training algorithms
already implemented require the entire dataset to be available in memory.

**Workaround**

* Downsample your dataset. A rule of thumb is that TF-DF training
uses 4 bytes per input dimension, so a dataset with 100 million examples and 10
numerical/categorical features would be 4 GB in memory.

* Train a manual ensemble on slices of the dataset, i.e. train N models on N
slices of data, and average the predictions.

## No support for GPU / TPU.

TF-DF does not support GPU or TPU training. Compiling with AVX instructions,
however, may speed up serving.

## No support for [model_to_estimator](https://www.tensorflow.org/api_docs/python/tf/keras/estimator/model_to_estimator)

TF-DF does not implement the APIs required to convert a trained/untrained model
to the estimator format.
