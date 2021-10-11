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

## Incompatibility with old or nightly version of TensorFlow

TensorFlow [ABI](https://en.wikipedia.org/wiki/Application_binary_interface) is
not compatible in between releases. Because TF-DF relies on custom TensorFlow
C++ ops, each version of TF-DF is tied to a specific version of TensorFlow. The
last released version of TF-DF is always tied to the last released version of
TensorFlow.

For reasons, the current version of TF-DF might not be compatible with older
versions or with the nightly build of TensorFlow.

If using incompatible versions of TF and TF-DF, you will see cryptic errors such
as:

```
tensorflow_decision_forests/tensorflow/ops/training/training.so: undefined symbol: _ZN10tensorflow11GetNodeAttrERKNS_9AttrSliceEN4absl14lts_2020_09_2311string_viewEPSs
```

**Workarounds:**

-   Use the version of TF-DF that is compatible with your version of TensorFlow.

### Compatibility table

The following table shows the compatibility between
`tensorflow_decision_forests` and its dependencies:

tensorflow_decision_forests | tensorflow
--------------------------- | ----------
0.1.9                       | 2.6
0.1.1 - 0.1.8               | 2.5
0.1.0                       | 2.4

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
