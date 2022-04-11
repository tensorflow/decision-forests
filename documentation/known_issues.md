# Known Issues

The underlying engine behind the decision forests algorithms used by TensorFlow
Decision Forests have been extensively production-tested. But this API to
TensorFlow and Keras is new, and some issues are expected -- we are trying to
fix them as quickly as possible.

See also the
[known issues of Yggdrasil Decision Forests](https://github.com/google/yggdrasil-decision-forests/blob/main/documentation/known_issues.md)
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

-   Use the version of TF-DF that is compatible with your version of TensorFlow.

### Compatibility table

The following table shows the compatibility between
`tensorflow_decision_forests` and its dependencies:

tensorflow_decision_forests | tensorflow
--------------------------- | ----------
0.2.4                       | 2.8
0.2.1 - 0.2.3               | 2.7
0.1.9 - 0.2.0               | 2.6
0.1.1 - 0.1.8               | 2.5
0.1.0                       | 2.4

-   *Solution #2:* Wrapps your preprocessing function into another function that
    [squeeze](https://www.tensorflow.org/api_docs/python/tf/squeeze) its inputs.

## No all models support distributed training and distribute strategies

Unless specified, models are trained on a single machine and are not compatible
with distribution strategies. For example the `GradientBoostedTreesModel` does
not support distributed training while `DistributedGradientBoostedTreesModel`
does.

**Workarounds:**

-   Use a model that support distribution strategies (e.g.
    `DistributedGradientBoostedTreesModel`), or downsample your dataset so it
    fits on a single machine.

## No support for GPU / TPU.

TF-DF does not support GPU or TPU training. Compiling with AVX instructions,
however, may speed up serving.

## No support for [model_to_estimator](https://www.tensorflow.org/api_docs/python/tf/keras/estimator/model_to_estimator)

TF-DF does not implement the APIs required to convert a trained/untrained model
to the estimator format.
