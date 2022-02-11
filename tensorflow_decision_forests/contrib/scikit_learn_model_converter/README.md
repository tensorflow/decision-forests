# Scikit-Learn Converter

## Introduction

**Scikit-Learn Model Converter** converts Scikit-Learn tree-based models (e.g.
Random Forests) into TensorFlow models compatible with the whole TensorFlow
ecosystem (e.g. Keras composition, SavedModel format, TF-Serving).

The converted model is also a TensorFlow Decision Forests (TF-DF) model
compatible with all of TF-DF functionalities (e.g. plotting, c++ inference API).

## Currently supported models

*   `sklearn.tree.DecisionTreeClassifier`
*   `sklearn.tree.DecisionTreeRegressor`
*   `sklearn.tree.ExtraTreeClassifier`
*   `sklearn.tree.ExtraTreeRegressor`
*   `sklearn.ensemble.RandomForestClassifier`
*   `sklearn.ensemble.RandomForestRegressor`
*   `sklearn.ensemble.ExtraTreesClassifier`
*   `sklearn.ensemble.ExtraTreesRegressor`
*   `sklearn.ensemble.GradientBoostingRegressor`

## Usage example

```python
from sklearn import datasets
from sklearn import tree
import tensorflow as tf
from tensorflow_decision_forests.contrib import scikit_learn_model_converter

# Train your model in scikit-learn
X, y = datasets.make_classification()
sklearn_model = tree.DecisionTreeClassifier().fit(X, y)

# Convert to tensorflow and predict
tensorflow_model = scikit_learn_model_converter.convert(sklearn_model)
y_pred = tensorflow_model.predict(tf.constant(X))
```
