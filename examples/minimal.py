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

"""Minimal usage example of TensorFlow Decision Forests.

This example trains, display and evaluate a Random Forest model on the adult
dataset.

This example works with the pip package.

Usage example (in a shell):

  pip3 install tensorflow_decision_forests
  python3 minimal.py

More examples are available in the documentation's colabs.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf

# Check the current version of TensorFlow Decision Forests
print("Found TF-DF v" + tfdf.__version__)

# Download and assemble the Adult dataset.
dataset_path = tf.keras.utils.get_file(
    "adult.csv",
    "https://raw.githubusercontent.com/google/yggdrasil-decision-forests/"
    "main/yggdrasil_decision_forests/test_data/dataset/adult.csv")

# Load a dataset into a Pandas Dataframe.
dataset_df = pd.read_csv(dataset_path)  # "df" for Pandas's DataFrame.

print("First 3 examples:")
print(dataset_df.head(3))
# Note that the dataset contains a mix of numerical and categorical features.
# TensorFlow Decision Forests will handle them automatically! (e.g. no see for
# one-hot encoding or normalization; except for the label).

# Split the dataset into a training and a testing dataset.
test_indices = np.random.rand(len(dataset_df)) < 0.30
test_ds_pd = dataset_df[test_indices]
train_ds_pd = dataset_df[~test_indices]
print(f"{len(train_ds_pd)} examples in training"
      f", {len(test_ds_pd)} examples for testing.")

# Converts a Pandas dataset into a tensorflow dataset
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label="income")
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label="income")

# Important: If you build the tf.dataset manually:
# - The batch size has no effect on the algorithm. 64 is a good default value.
# - Don't shuffle the training dataset (unlike Neural Net, the algorithm does
#   not benefit from shuffling). Bonus: The algorithm is deterministic (running
#   it twice on the same dataset will give the same model).
# - Don't use "repeats". The dataset should contain exactly one epoch.

# Trains the model.
model = tfdf.keras.RandomForestModel(verbose=2)
model.fit(x=train_ds)

# Note: If running in a Colab, ".fit()" will not print the training logs by
# default. To do so, you need to encapsulate the "fit()" in a
# "from wurlitzer import sys_pipescall" (see the colabs for some examples).

# Some information about the model.
# Different learning algorithm (and different hyper-parameters) can output
# different information.
print(model.summary())

# Evaluate the model on the validation dataset.
model.compile(metrics=["accuracy"])
evaluation = model.evaluate(test_ds)

# The first entry is the BinaryCrossentropy loss. The next entries are specified
# by compile's metrics.
print(f"BinaryCrossentropyloss: {evaluation[0]}")
print(f"Accuracy: {evaluation[1]}")

# Export the model to the SavedModel format for later re-use e.g. TensorFlow
# Serving.
model.save("/tmp/my_saved_model")

# Note: This model is compatible with Yggdrasil Decision Forests.

# Look at the feature importances.
model.make_inspector().variable_importances()
