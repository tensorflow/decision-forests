# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tools for converting scikit-learn tree-based models to TFDF models.

This module converts Scikit-Learn tree-based models (e.g.
Random Forests) into TensorFlow models compatible with the whole TensorFlow
ecosystem (e.g. Keras composition, SavedModel format, TF-Serving).

The converted model is also a TensorFlow Decision Forests (TF-DF) model
compatible with all of TF-DF functionalities (e.g. plotting, c++ inference API).

Example usage:

```
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

"""

from tensorflow_decision_forests.contrib.scikit_learn_model_converter import scikit_learn_model_converter as lib

convert = lib.convert
convert_sklearn_tree_to_tfdf_pytree = lib.convert_sklearn_tree_to_tfdf_pytree
