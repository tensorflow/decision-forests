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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
from absl import logging
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf

from google.protobuf import text_format


def data_root_path() -> str:
  return ""


def test_data_path() -> str:
  return os.path.join(data_root_path(),
                      "external/ydf/yggdrasil_decision_forests/test_data")


def tmp_path() -> str:
  return flags.FLAGS.test_tmpdir


class TFDFTunerTest(tf.test.TestCase):

  def test_random_adult_in_memory(self):

    # Prepare the datasets
    dataset_directory = os.path.join(test_data_path(), "dataset")
    train_path = os.path.join(dataset_directory, "adult_train.csv")
    test_path = os.path.join(dataset_directory, "adult_test.csv")

    label = "income"

    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
        pd.read_csv(train_path), label=label)
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
        pd.read_csv(test_path), label=label)

    # Configure and train the model
    tuner = tfdf.tuner.RandomSearch(num_trials=30)
    tuner.choice("num_candidate_attributes_ratio", [1.0, 0.8, 0.6])
    tuner.choice("use_hessian_gain", [True, False])

    local_search_space = tuner.choice("growing_strategy", ["LOCAL"])
    local_search_space.choice("max_depth", [4, 5, 6, 7])

    global_search_space = tuner.choice(
        "growing_strategy", ["BEST_FIRST_GLOBAL"], merge=True)
    global_search_space.choice("max_num_nodes", [16, 32, 64, 128])

    model = tfdf.keras.GradientBoostedTreesModel(num_trees=50, tuner=tuner)
    model.fit(train_ds)

    # Evaluate the model
    model.compile(["accuracy"])
    evaluation = model.evaluate(test_ds, return_dict=True)
    self.assertGreater(evaluation["accuracy"], 0.87)

    tuning_logs = model.make_inspector().tuning_logs()
    logging.info("Tuning logs:\n%s", tuning_logs)

    self.assertSetEqual(
        set(tuning_logs.columns),
        set([
            "score", "evaluation_time", "best",
            "num_candidate_attributes_ratio", "use_hessian_gain",
            "growing_strategy", "max_depth", "max_num_nodes"
        ]))
    self.assertEqual(tuning_logs.shape, (30, 8))
    self.assertEqual(tuning_logs["best"].sum(), 1)
    self.assertNear(tuning_logs["score"][tuning_logs["best"]].values[0], -0.587,
                    0.05)

    # This is a lot of text.
    _ = model.make_inspector().tuning_logs(return_format="proto")

  def test_random_adult_in_memory_predefined_hpspace(self):

    # Prepare the datasets
    dataset_directory = os.path.join(test_data_path(), "dataset")
    train_path = os.path.join(dataset_directory, "adult_train.csv")
    test_path = os.path.join(dataset_directory, "adult_test.csv")

    label = "income"

    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
        pd.read_csv(train_path), label=label)
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
        pd.read_csv(test_path), label=label)

    # Configure and train the model
    tuner = tfdf.tuner.RandomSearch(num_trials=30, use_predefined_hps=True)
    model = tfdf.keras.GradientBoostedTreesModel(num_trees=50, tuner=tuner)
    model.fit(train_ds)

    # Evaluate the model
    model.compile(["accuracy"])
    evaluation = model.evaluate(test_ds, return_dict=True)
    self.assertGreater(evaluation["accuracy"], 0.87)

    tuning_logs = model.make_inspector().tuning_logs()
    logging.info("Tuning logs:\n%s", tuning_logs)

    self.assertSetEqual(
        set(tuning_logs.columns),
        set([
            "score", "evaluation_time", "best",
            "num_candidate_attributes_ratio", "use_hessian_gain",
            "growing_strategy", "max_depth", "max_num_nodes", "subsample",
            "shrinkage", "sampling_method", "sparse_oblique_weights",
            "sparse_oblique_projection_density_factor", "categorical_algorithm",
            "min_examples", "sparse_oblique_normalization", "split_axis"
        ]))
    self.assertEqual(tuning_logs.shape, (30, 17))
    self.assertEqual(tuning_logs["best"].sum(), 1)
    self.assertNear(tuning_logs["score"][tuning_logs["best"]].values[0], -0.587,
                    0.05)

    # This is a lot of text.
    _ = model.make_inspector().tuning_logs(return_format="proto")


if __name__ == "__main__":
  tf.test.main()
