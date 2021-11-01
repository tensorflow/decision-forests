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
from typing import List

from absl import flags
from absl import logging
from absl.testing import parameterized
import numpy as np
import pandas as pd
import portpicker
import tensorflow as tf

from tensorflow.python.distribute import distribute_lib
import tensorflow_decision_forests as tfdf


def data_root_path() -> str:
  return ""


def test_data_path() -> str:
  return os.path.join(data_root_path(),
                      "external/ydf/yggdrasil_decision_forests/test_data")


def tmp_path() -> str:
  return flags.FLAGS.test_tmpdir


def _create_in_process_cluster(num_workers, num_ps):
  """Create a cluster of TF workers and returns their addresses.

  Such cluster simulate the behavior of multiple TF parameter servers.

  Args:
    num_workers: Number of "worker" workers.
    num_ps: Number of "parameter server" workers.

  Returns:
    The ClusterResolver i.e. the ip addresses of the workers.
  """

  worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

  cluster_dict = {"worker": ["localhost:%s" % port for port in worker_ports]}
  if num_ps > 0:
    cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]

  cluster_spec = tf.train.ClusterSpec(cluster_dict)
  worker_config = tf.compat.v1.ConfigProto()

  for i in range(num_workers):
    tf.distribute.Server(
        cluster_spec,
        job_name="worker",
        task_index=i,
        config=worker_config,
        protocol="grpc")

  for i in range(num_ps):
    tf.distribute.Server(
        cluster_spec, job_name="ps", task_index=i, protocol="grpc")

  os.environ["GRPC_FAIL_FAST"] = "use_caller"

  return tf.distribute.cluster_resolver.SimpleClusterResolver(
      cluster_spec, rpc_layer="grpc")


class TFDFDistributedTest(parameterized.TestCase, tf.test.TestCase):

  def test_distributed_training_synthetic(self):
    # Based on
    # https://www.tensorflow.org/tutorials/distribute/parameter_server_training

    # Create a distributed dataset.
    global_batch_size = 20
    num_examples = 1000
    num_features = 2

    def make_dataset(seed):
      x = tf.random.uniform((num_examples, num_features), seed=seed)
      y = x[:, 0] + (x[:, 1] - 0.5) * 0.1 >= 0.5
      return tf.data.Dataset.from_tensor_slices((x, y))

    def dataset_fn(context: distribute_lib.InputContext, seed: int):
      dataset = make_dataset(seed=seed)

      if context is not None:
        # Split the dataset among the workers.
        current_worker = tfdf.keras.get_worker_idx_and_num_workers(context)
        assert current_worker.num_workers == 4
        dataset = dataset.shard(
            num_shards=current_worker.num_workers,
            index=current_worker.worker_idx)

      # Currently, if any of the workers runs out of training examples, the
      # keras training will either fail or raise a wall of warnings and
      # continue.
      #
      # TODO(gbm): Remove repeat when possible.
      dataset = dataset.repeat(None)

      dataset = dataset.batch(global_batch_size)
      dataset = dataset.prefetch(2)
      return dataset

    # Create the workers
    cluster_resolver = _create_in_process_cluster(num_workers=4, num_ps=1)

    # Configure the model and datasets
    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver)
    with strategy.scope():
      model = tfdf.keras.DistributedGradientBoostedTreesModel()
      model.compile(metrics=["accuracy"])
      # Note: "tf.keras.utils.experimental.DatasetCreator" seems to also work.
      train_dataset_creator = strategy.distribute_datasets_from_function(
          lambda context: dataset_fn(context, 111))
      valid_dataset_creator = strategy.distribute_datasets_from_function(
          lambda context: dataset_fn(context, 222))
      # Note: A distributed dataset cannot be reused twice.
      valid_dataset_creator_again = strategy.distribute_datasets_from_function(
          lambda context: dataset_fn(context, 333))

    # Train model
    training_history = model.fit(
        train_dataset_creator,
        steps_per_epoch=num_examples // global_batch_size,
        validation_data=valid_dataset_creator,
        validation_steps=num_examples / global_batch_size)
    logging.info("Training history: %s", training_history.history)
    self.assertGreaterEqual(training_history.history["val_accuracy"][0], 0.98)

    # Re-evaluate the model on a validation dataset.
    valid_evaluation_again = model.evaluate(
        valid_dataset_creator_again,
        steps=num_examples // global_batch_size,
        return_dict=True)
    logging.info("Valid evaluation (again): %s", valid_evaluation_again)
    self.assertGreaterEqual(valid_evaluation_again["accuracy"], 0.99)

    logging.info("Trained model:")
    model.summary()

    # Check the models structure.
    inspector = model.make_inspector()
    self.assertEqual([f.name for f in inspector.features()],
                     ["data:0.0", "data:0.1"])
    self.assertEqual(inspector.label().name, "__LABEL")
    self.assertEqual(inspector.num_trees(), 300)

  def test_distribution_strategy_not_supported(self):

    # Create a distributed dataset.
    global_batch_size = 20
    num_examples = 200
    num_features = 2

    def make_dataset(seed):
      x = tf.random.uniform((num_examples, num_features), seed=seed)
      y = x[:, 0] + (x[:, 1] - 0.5) * 0.1 >= 0.5
      return tf.data.Dataset.from_tensor_slices((x, y))

    def dataset_fn(input_context):
      batch_size = input_context.get_per_replica_batch_size(global_batch_size)
      dataset = make_dataset(seed=input_context.input_pipeline_id)
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)
      dataset = dataset.batch(batch_size)
      dataset = dataset.prefetch(2)
      return dataset

    dataset_creator = tf.keras.utils.experimental.DatasetCreator(dataset_fn)

    cluster_resolver = _create_in_process_cluster(num_workers=2, num_ps=1)

    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver)
    with strategy.scope():
      model = tfdf.keras.GradientBoostedTreesModel()

    with self.assertRaisesRegex(
        ValueError,
        "does not support training with a TF Distribution strategy"):
      model.fit(
          dataset_creator, steps_per_epoch=num_examples / global_batch_size)

  def _shard_dataset(self, path, num_shards=20) -> List[str]:
    """Splits a csv dataset into multiple csv files."""

    dataset = pd.read_csv(path)
    split_datasets = np.array_split(dataset, num_shards)
    output_dir = self.get_temp_dir()
    output_paths = []
    for i, d in enumerate(split_datasets):
      output_path = os.path.join(output_dir, f"dataset-{i}-of-{num_shards}.csv")
      output_paths.append(output_path)
      d.to_csv(output_path, index=False)
    return output_paths

  def test_distributed_training_adult(self):

    # Shard the dataset.
    train_path = os.path.join(test_data_path(), "dataset", "adult_train.csv")
    test_path = os.path.join(test_data_path(), "dataset", "adult_test.csv")

    sharded_train_paths = self._shard_dataset(train_path)
    logging.info("Num sharded paths: %d", len(sharded_train_paths))

    global_batch_size = 20
    num_train_examples = pd.read_csv(train_path).shape[0]
    num_test_examples = pd.read_csv(test_path).shape[0]
    logging.info("num_train_examples: %d", num_train_examples)
    logging.info("num_test_examples: %d", num_test_examples)

    # Create the dataset
    def dataset_fn(context: distribute_lib.InputContext, paths, infinite=False):
      logging.info("Create dataset with context: %s and %d path(s)", context,
                   len(paths))

      ds_path = tf.data.Dataset.from_tensor_slices(paths)

      if context is not None:
        # Split the dataset among the workers.
        current_worker = tfdf.keras.get_worker_idx_and_num_workers(context)
        assert current_worker.num_workers == 5
        ds_path = ds_path.shard(
            num_shards=current_worker.num_workers,
            index=current_worker.worker_idx)

      if infinite:
        ds_path = ds_path.repeat(None)

      def read_csv_file(path):
        numerical = tf.constant([0.0], dtype=tf.float32)
        categorical_string = tf.constant(["NA"], dtype=tf.string)
        csv_columns = [
            numerical,  # age
            categorical_string,  # workclass
            numerical,  # fnlwgt
            categorical_string,  # education
            numerical,  # education_num
            categorical_string,  # marital_status
            categorical_string,  # occupation
            categorical_string,  # relationship
            categorical_string,  # race
            categorical_string,  # sex
            numerical,  # capital_gain
            numerical,  # capital_loss
            numerical,  # hours_per_week
            categorical_string,  # native_country
            categorical_string,  # income
        ]
        return tf.data.experimental.CsvDataset(path, csv_columns, header=True)

      ds_columns = ds_path.interleave(read_csv_file)

      init_label_table = tf.lookup.KeyValueTensorInitializer(
          keys=tf.constant(["<=50K", ">50K"]),
          values=tf.constant([0, 1], dtype=tf.int64))
      label_table = tf.lookup.StaticVocabularyTable(
          init_label_table, num_oov_buckets=1)

      def extract_label(*columns):
        return columns[0:-1], label_table.lookup(columns[-1])

      ds_dataset = ds_columns.map(extract_label)
      ds_dataset = ds_dataset.batch(global_batch_size)
      return ds_dataset

    # Create the workers
    cluster_resolver = _create_in_process_cluster(num_workers=5, num_ps=1)

    # Configure the model and datasets
    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver)

    with strategy.scope():
      model = tfdf.keras.DistributedGradientBoostedTreesModel()
      model.compile(metrics=["accuracy"])

      # Currently, if any of the workers runs out of training examples, the
      # keras training will either fail or raise a wall of warnings and
      # continue.
      #
      # TODO(gbm): Remove infinite=True when possible.
      train_dataset = strategy.distribute_datasets_from_function(
          lambda context: dataset_fn(
              context, sharded_train_paths, infinite=True))

    # Train model
    #
    # Currently, keras requires the number of training steps in the case of
    # distributed training.
    #
    # TODO(gbm): Figure a way to use finite datasets in distributed training
    # e.g. steps_per_epoch=-1 (?)
    model.fit(
        train_dataset,
        steps_per_epoch=num_train_examples // global_batch_size,
    )

    logging.info("Trained model:")
    model.summary()

    # Non-distributed evaluation of the model.
    model._distribution_strategy = None
    model._cluster_coordinator = None
    evaluation = model.evaluate(dataset_fn(None, [test_path]), return_dict=True)
    logging.info("Evaluation: %s", evaluation)

    # If all the workers are running at the same speed, at most
    # "(global_batch_size-1) * num_workers examples" can be lost before
    # training due to the Keras limitation discussed in the notes above.
    # However, because of the currently required repeat, if workers are running
    # at different speed, some examples can be repeated.
    self.assertAlmostEqual(evaluation["accuracy"], 0.8603476, delta=0.02)

  def test_distributed_training_adult_from_disk(self):
    # Path to dataset.
    dataset_directory = os.path.join(test_data_path(), "dataset")
    train_path = os.path.join(dataset_directory, "adult_train.csv")
    test_path = os.path.join(dataset_directory, "adult_test.csv")

    label = "income"

    # Create the workers
    cluster_resolver = _create_in_process_cluster(num_workers=5, num_ps=1)

    # Configure the model and datasets
    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver)

    with strategy.scope():
      model = tfdf.keras.DistributedGradientBoostedTreesModel()
      model.compile(metrics=["accuracy"])

    training_history = model.fit_on_dataset_path(
        train_path=train_path,
        label_key=label,
        dataset_format="csv",
        valid_path=test_path)
    logging.info("Training history: %s", training_history.history)

    logging.info("Trained model:")
    model.summary()

    model._distribution_strategy = None
    test_df = pd.read_csv(test_path)
    tf_test = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label)
    evaluation = model.evaluate(tf_test, return_dict=True)
    logging.info("Evaluation: %s", evaluation)
    self.assertAlmostEqual(evaluation["accuracy"], 0.8703476, delta=0.01)

    features = [feature.name for feature in model.make_inspector().features()]
    self.assertEqual(features, [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country"
    ])

  def test_in_memory_not_supported(self):

    dataframe = pd.DataFrame({
        "a b": [0, 1, 2],
        "c,d": [0, 1, 2],
        "e%f": [0, 1, 2],
        "a%b": [0, 1, 2],
        "label": [0, 1, 2]
    })
    dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataframe, label="label")

    model = tfdf.keras.DistributedGradientBoostedTreesModel()

    with self.assertRaisesRegex(
        tf.errors.UnknownError,
        "does not support training from in-memory datasets"):
      model.fit(dataset)


if __name__ == "__main__":
  tf.test.main()
