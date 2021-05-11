# Known Issues

The underlying engine behind the decision forests algorithms used by TensorFlow
Decision Forests have been extensively production-tested. But this API to
TensorFlow and Keras is new, and some issues are expected -- we are trying to
fix them as quickly as possible.

See also the
[known issues of Yggdrasil Decision Forests](https://github.com/google/yggdrasil-decision-forests/documentation/known_issues.md).

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
