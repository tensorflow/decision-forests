<p align="center">
<img src="documentation/image/logo.png"  />
</p>

**TensorFlow Decision Forests** (**TF-DF**) is a library to train, run and
interpret [decision forest](https://ydf.readthedocs.io/en/latest/intro_df.html)
models (e.g., Random Forests, Gradient Boosted Trees) in TensorFlow. TF-DF
supports classification, regression and ranking.

**TF-DF** is powered by
[Yggdrasil Decision Forest](https://github.com/google/yggdrasil-decision-forests)
(**YDF**, a library to train and use decision forests in C++, JavaScript, CLI,
and Go. TF-DF models are
[compatible](https://ydf.readthedocs.io/en/latest/convert_model.html#convert-a-a-tensorflow-decision-forests-model-to-a-yggdrasil-model)
with YDF' models, and vice versa.

Tensorflow Decision Forests is available on Linux and Mac. Windows users can use
the library through WSL+Linux.

## Usage example

A minimal end-to-end run looks as follows:

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

## Google IO Presentation

<div align="center">
    <a href="https://youtu.be/5qgk9QJ4rdQ" target="Video">
        <img src="https://img.youtube.com/vi/5qgk9QJ4rdQ/0.jpg"></img>
    </a>
</div>


## Documentation & Resources

The following resources are available:

-   [TF-DF on TensorFlow.org](https://tensorflow.org/decision_forests) (API
    Reference, Guides and Tutorials)
-   [Tutorials](https://www.tensorflow.org/decision_forests/tutorials) (on
    tensorflow.org)
-   [YDF documentation](https://ydf.readthedocs.io) (also applicable to TF-DF)
-   [Issue tracker](https://github.com/tensorflow/decision-forests/issues)
-   [Known issues](documentation/known_issues.md)
-   [Changelog](CHANGELOG.md)
-   [TensorFlow Forum](https://discuss.tensorflow.org) (on
    discuss.tensorflow.org)
-   [More examples](documentation/more_examples.md)

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
-   Richard Stotz (richardstotz AT google DOT com)
-   Sebastian Bruch (sebastian AT bruch DOT io)
-   Arvind Srinivasan (arvnd AT google DOT com)

## License

[Apache License 2.0](LICENSE)
