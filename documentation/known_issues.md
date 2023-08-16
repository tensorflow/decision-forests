# Known Issues

The underlying engine behind the decision forests algorithms used by TensorFlow
Decision Forests have been extensively production-tested. This file lists some
of the known issues.

See also the [migration guide](migration.md) for behavior that is different from
other algorithms.

## Windows Pip package is not available

TensorFlow Decision Forests is not yet available as a Windows Pip package.

**Workarounds:**

-   *Solution #1:* Install
    [Windows Subsystem for Linux (WSL)](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux)
    on your Windows machine and follow the Linux instructions.

## Incompatibility with old or nightly versions of TensorFlow

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
--------------------------- | ---------------
1.5.0                       | 2.13.0
1.3.0 - 1.4.0               | 2.12.0
1.1.0 - 1.2.0               | 2.11.0
1.0.0 - 1.0.1               | 2.10.0 - 2.10.1
0.2.6 - 0.2.7               | 2.9.1
0.2.5                       | 2.9
0.2.4                       | 2.8
0.2.1 - 0.2.3               | 2.7
0.1.9 - 0.2.0               | 2.6
0.1.1 - 0.1.8               | 2.5
0.1.0                       | 2.4

-   *Solution #2:* Wrap your preprocessing function into another function that
    [squeezes](https://www.tensorflow.org/api_docs/python/tf/squeeze) its
    inputs.

## Not all models support distributed training and distribute strategies

Unless specified, models are trained on a single machine and are not compatible
with distribution strategies. For example the `GradientBoostedTreesModel` does
not support distributed training while `DistributedGradientBoostedTreesModel`
does.

**Workarounds:**

-   Use a model that support distribution strategies (e.g.
    `DistributedGradientBoostedTreesModel`), or downsample your dataset so that
    it fits on a single machine.

## No support for GPU / TPU.

TF-DF does not supports GPU or TPU training. Compiling with AVX instructions,
however, may speed up serving.

## No support for [model_to_estimator](https://www.tensorflow.org/api_docs/python/tf/keras/estimator/model_to_estimator)

TF-DF does not implement the APIs required to convert a trained/untrained model
to the estimator format.

## Loaded models behave differently than Python models.

While abstracted by the Keras API, a model instantiated in Python (e.g., with
`tfdf.keras.RandomForestModel()`) and a model loaded from disk (e.g., with
`tf.keras.models.load_model()`) can behave differently. Notably, a Python
instantiated model automatically applies necessary type conversions. For
example, if a `float64` feature is fed to a model expecting a `float32` feature,
this conversion is performed implicitly. However, such a conversion is not
possible for models loaded from disk. It is therefore important that the
training data and the inference data always have the exact same type.

## Tensorflow feature name sanitization

Tensorflow sanitizes feature names and might, for instance, convert them to
lowercase.
