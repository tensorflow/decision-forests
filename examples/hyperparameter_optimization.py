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

r"""Example of automated hyper-parameter tuning with TensorFlow Decision Forests.

This example trains, displays, evaluates and export a Gradient Boosted Tree
model.

Usage example:

  pip3 install tensorflow_decision_forests -U
  python3 hyperparameter_optimization.py

Or

  bazel run -c opt \
  //tensorflow_decision_forests/examples:hyperparameter_optimization
  \
  -- --alsologtostderr
"""

from absl import app

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Download the Adult dataset.
  dataset_path = tf.keras.utils.get_file(
      "adult.csv",
      "https://raw.githubusercontent.com/google/yggdrasil-decision-forests/"
      "main/yggdrasil_decision_forests/test_data/dataset/adult.csv")

  # Load a dataset into a Pandas Dataframe.
  dataset_df = pd.read_csv(dataset_path)  # "df" for Pandas's DataFrame.

  print("First the first three examples:")
  print(dataset_df.head(3))

  # Notice that the dataset contains a mix of numerical and categorical
  # features. TensorFlow Decision Forests handles them automatically (e.g. no
  # need for one-hot encoding or normalization; except for the label).

  # Split the dataset into a training and a testing dataset.
  test_indices = np.random.rand(len(dataset_df)) < 0.30
  test_ds_pd = dataset_df[test_indices]
  train_ds_pd = dataset_df[~test_indices]
  print(f"{len(train_ds_pd)} examples in training"
        f", {len(test_ds_pd)} examples for testing.")

  # Converts datasets from Pandas dataframe to TensorFlow dataset format.
  train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label="income")
  test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label="income")

  # Tune the model.
  #
  # The hyper-parameters to optimize are automatically set with
  # "use_predefined_hps=True". See
  # https://www.tensorflow.org/decision_forests/tutorials/automatic_tuning_colab
  # for an example where the hyper-parameter space is configured manually.
  tuner = tfdf.tuner.RandomSearch(num_trials=30, use_predefined_hps=True)
  model = tfdf.keras.GradientBoostedTreesModel(verbose=2, tuner=tuner)
  model.fit(train_ds)

  # Some information about the model.
  print(model.summary())

  # Evaluates the model on the test dataset.
  model.compile(metrics=["accuracy"])
  evaluation = model.evaluate(test_ds)
  print(f"BinaryCrossentropyloss: {evaluation[0]}")
  print(f"Accuracy: {evaluation[1]}")

  # Exports the model to disk in the SavedModel format for later re-use. This
  # model can be used with TensorFlow Serving and Yggdrasil Decision Forests
  # (https://ydf.readthedocs.io/en/latest/serving_apis.html).
  model.save("/tmp/my_saved_model")


if __name__ == "__main__":
  app.run(main)
