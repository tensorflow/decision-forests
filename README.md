# TensorFlow Decision Forests

[![TensorFlow](https://img.shields.io/badge/TensorFow-page-orange)](https://www.tensorflow.org/decision_forests)

**TensorFlow Decision Forests** (**TF-DF**) is a collection of state-of-the-art
algorithms for the training, serving and interpretation of **Decision Forest**
models. The library is a collection of [Keras](https://keras.io/) models and
supports classification, regression and ranking.

**TF-DF** is a [TensorFlow](https://www.tensorflow.org/) wrapper around the
[Yggdrasil Decision Forests](https://github.com/google/yggdrasil-decision-forests)
C++ libraries. Models trained with TF-DF are compatible with Yggdrasil Decision
Forests' models, and vice versa.

## Usage example

A minimal end-to-end run looks as follow:

```python
import tensorflow_decision_forests as tfdf
import pandas as pd

# Load the dataset in a Pandas dataframe.
train_df = pd.read_csv("project/train.csv")
test_df = pd.read_csv("project/test.csv")

# Convert the dataset into a TensorFlow dataset.
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="my_label")
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="my_label")

# Train the model
model = tfdf.keras.RandomForestModel()
model.fit(train_ds)

# Look at the model.
model.summary()

# Evaluate the model.
model.evaluate(test_ds)

# Export to a TensorFlow SavedModel.
# Note: the model is compatible with Yggdrasil Decision Forests.
model.save("project/model")
```

## Documentation & Resources

The following resources are available:

-   [TF-DF on TensorFlow.org](https://tensorflow.org/decision_forests) (with the
    API Reference and Tutorials)
-   [Colabs](documentation/tutorials)
-   [Migration guide from Neural Network to Decision Forests](documentation/migration.md)
-   [Issue tracker](https://github.com/tensorflow/decision-forests/issues)
-   [Known issues](documentation/known_issues.md)
-   [Changelog](CHANGELOG.md)
-   [Discuss on TensorFlow.Org](https://discuss.tensorflow.org)
-   [Yggdrasil documentation](https://github.com/google/yggdrasil-decision-forests)
    (for advanced users and C++ serving)
-   [Tutorials](https://www.tensorflow.org/decision_forests/tutorials)
-   [TF-DF in TensorFlow Serving](documentation/tensorflow_serving)

## Installation

To install TensorFlow Decision Forests, run:

```shell
pip3 install tensorflow_decision_forests --upgrade
```

See the [installation](documentation/installation.md) page for more details,
troubleshooting and alternative installation solutions.

## Contributing

Contributions to TensorFlow Decision Forests and Yggdrasil Decision Forests are
welcome. If you want to contribute, make sure to review the
[developer manual](documentation/developer_manual.md) and
[contribution guidelines](CONTRIBUTING.md).

## Credits

TensorFlow Decision Forests was developed by:

-   Mathieu Guillame-Bert (gbm AT google DOT com)
-   Jan Pfeifer (janpf AT google DOT com)
-   Sebastian Bruch (sebastian AT bruch DOT io)
-   Arvind Srinivasan (arvnd AT google DOT com)

## License

[Apache License 2.0](LICENSE)
