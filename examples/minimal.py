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
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf

# Check the current version of TensorFlow Decision Forests
print("Found TF-DF v" + tfdf.__version__)

# Download and assemble the Adult dataset.
print("Assemble dataset")
dataset_without_header_path = tf.keras.utils.get_file(
    "adult.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
header = ("age,workclass,fnlwgt,education,education_num,marital_status,"
          "occupation,relationship,race,sex,capital_gain,capital_loss,"
          "hours_per_week,native_country,income")
dataset_path = os.path.join(
    os.path.dirname(dataset_without_header_path), "adult.csv")

with open(dataset_path, "w") as f:
  f.write(header + "\n")
  f.write(open(dataset_without_header_path).read())

# Load a dataset into a Pandas Dataframe.
dataset_df = pd.read_csv(dataset_path)  # "df" for Pandas's DataFrame.

print("First 3 examples:")
print(dataset_df.head(3))
# Note that the dataset contains a mix of numerical and categorical features.
# TensorFlow Decision Forests will handle them automatically! (e.g. no see for
# one-hot encoding or normalization; except for the label).

# Remove the "float(NaN)" values from the categorical/string features.
for col in dataset_df.columns:
  if dataset_df[col].dtype in [str, object]:
    dataset_df[col] = dataset_df[col].fillna("")

# Name of the label column.
label = "income"

# Encode the categorical label into an integer.
# Note: Keras expected classification labels to be integers. However, you
# don't need to encode the features.
classes = dataset_df[label].unique().tolist()
print(f"Label classes: {classes}")

dataset_df[label] = dataset_df[label].map(classes.index)
print("First 3 examples with integer labels:")
print(dataset_df.head(3))

# Split the dataset into a training and a testing dataset.
test_indices = np.random.rand(len(dataset_df)) < 0.30
test_ds_pd = dataset_df[test_indices]
train_ds_pd = dataset_df[~test_indices]
print(
    f"{len(train_ds_pd)} examples in training, {len(test_ds_pd)} examples for testing."
)


# Converts a Pandas dataset into a tensorflow dataset
def df_to_ds(df):
  return tf.data.Dataset.from_tensor_slices(
      (dict(df.drop(label, 1)), df[label].values))


train_ds = df_to_ds(train_ds_pd).batch(64)
test_ds = df_to_ds(test_ds_pd).batch(64)

# Important:
# - The batch size has no effect on the algorithm.
# - Don't shuffle the training dataset (unlike Neural Net, the algorithm does
#   not benefit from shuffling). Bonus: The algorithm is deterministic (running
#   it twice on the same dataset will give the same model).
# - Don't use "repeats". The dataset should contain exactly one epoch.
model = tfdf.keras.RandomForestModel()

# Optionally set some evaluation metrics.
model.compile(metrics=["accuracy"])

# Trains the model.
model.fit(x=train_ds)

# Note: If running in a Colab, wrap the ".fit()" call in
# "with googlelog.CaptureLog()" to display the training logs.

# Some information about the model.
# Different learning algorithm (and different hyper-parameters) can output
# different information.
print(model.summary())

# Evaluate the model on the validation dataset.
evaluation = model.evaluate(test_ds)

# The first entry is the BinaryCrossentropy loss, the second is the accuracy.
# More metrics can be added in the "metrics" arguments of the model compilation.
print(f"BinaryCrossentropyloss: {evaluation[0]}")
print(f"Accuracy: {evaluation[1]}")
