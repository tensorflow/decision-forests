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

"""User entry point for the TensorFlow Decision Forest API.

Basic usage:

```
# Imports
import tensorflow_decision_forests as tfdf
import pandas as pd
from wurlitzer import sys_pipes

# Load a dataset into a Pandas Dataframe.
dataset_df = pd.read_csv("/tmp/penguins.csv")

# Display the first 3 examples.
dataset_df.head(3)

# Convert the Pandas dataframe to a tf dataset
tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset_df,label="species")

model = tfdf.keras.RandomForestModel()
with sys_pipes():
  model.fit(tf_dataset)
# Note: The `sys_pipes` part is to display logs during training.

# Evaluate model.
model.compile(metrics=["accuracy"])

# Save model.
model.save("/tmp/my_saved_model")

# ...

# Load a model: it loads as a generic keras model.
loaded_model = tf.keras.models.load_model("/tmp/my_saved_model")
```

"""

__version__ = "0.2.4"
__author__ = "Mathieu Guillame-Bert"

compatible_tf_versions = ["2.8.0"]

from tensorflow_decision_forests.tensorflow import check_version

check_version.check_version(__version__, compatible_tf_versions)

from tensorflow_decision_forests import keras
from tensorflow_decision_forests.component import py_tree
from tensorflow_decision_forests.component.builder import builder
from tensorflow_decision_forests.component.inspector import inspector
from tensorflow_decision_forests.component.model_plotter import model_plotter
