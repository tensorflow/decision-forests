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
import subprocess
import time
from typing import List, Tuple
import unittest

from absl import flags
from absl import logging
from absl.testing import parameterized
import numpy as np
import pandas as pd
import portpicker
import tensorflow as tf

from tensorflow.python.distribute import distribute_lib
import tensorflow_decision_forests as tfdf
from tensorflow_decision_forests.keras import keras_internal
from yggdrasil_decision_forests.learner.distributed_gradient_boosted_trees import distributed_gradient_boosted_trees_pb2


def data_root_path() -> str:
  return ""


def test_data_path() -> str:
  return os.path.join(
      data_root_path(), "external/ydf/yggdrasil_decision_forests/test_data"
  )


def tmp_path() -> str:
  return flags.FLAGS.test_tmpdir


def _create_in_process_tf_ps_cluster(num_workers, num_ps):
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
    try:
      tf.distribute.Server(
          cluster_spec,
          job_name="worker",
          task_index=i,
          config=worker_config,
          protocol="grpc",
      )
    except tf.errors.UnknownError as e:
      if "Could not start gRPC server" in e.message:
        raise unittest.SkipTest("Cannot start std servers.")
      else:
        raise

  for i in range(num_ps):
    try:
      tf.distribute.Server(
          cluster_spec, job_name="ps", task_index=i, protocol="grpc"
      )
    except tf.errors.UnknownError as e:
      if "Could not start gRPC server" in e.message:
        raise unittest.SkipTest("Cannot start std servers.")
      else:
        raise

  os.environ["GRPC_FAIL_FAST"] = "use_caller"

  return tf.distribute.cluster_resolver.SimpleClusterResolver(
      cluster_spec, rpc_layer="grpc"
  )


def _create_in_process_grpc_worker_cluster(
    num_workers,
) -> List[Tuple[str, int]]:
  """Create a cluster of GRPC workers and returns their addresses.

  Args:
    num_workers: Number of workers..

  Returns:
    List of socket addresses.
  """

  worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  worker_ip = "localhost"
  worker_addresses = []

  for i in range(num_workers):
    worker_addresses.append((worker_ip, worker_ports[i]))
    args = [
        "tensorflow_decision_forests/keras/grpc_worker_main",
        "--alsologtostderr",
        "--port",
        str(worker_ports[i]),
    ]
    subprocess.Popen(args, stdout=subprocess.PIPE)
    time.sleep(1)

  time.sleep(5)
  return worker_addresses


class TFDFDistributedTest(parameterized.TestCase, tf.test.TestCase):

  def test_distributed_training_synthetic(self):
    # Based on
    # https://www.tensorflow.org/tutorials/distribute/parameter_server_training

    # Create a distributed dataset.
    batch_size = 10
    num_examples = 1000
    num_features = 2

    def make_dataset(seed):
      x = tf.random.uniform((num_examples, num_features), seed=seed)
      y = x[:, 0] + (x[:, 1] - 0.5) * 0.1 >= 0.5
      return tf.data.Dataset.from_tensor_slices((x, y))

    def dataset_fn(
        context: distribute_lib.InputContext, seed: int, infinite: bool = False
    ) -> tf.data.Dataset:
      dataset = make_dataset(seed=seed)

      if context is not None:
        # Split the dataset among the workers.
        current_worker = tfdf.keras.get_worker_idx_and_num_workers(context)
        assert current_worker.num_workers == 4
        dataset = dataset.shard(
            num_shards=current_worker.num_workers,
            index=current_worker.worker_idx,
        )

      # TODO: Remove repeat when possible.
      if infinite:
        dataset = dataset.repeat(None)

      dataset = dataset.batch(batch_size)
      return dataset

    # Create the workers
    cluster_resolver = _create_in_process_tf_ps_cluster(num_workers=4, num_ps=1)

    # Configure the model and datasets
    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver
    )
    with strategy.scope():
      model = tfdf.keras.DistributedGradientBoostedTreesModel(worker_logs=False)
      model.compile(metrics=["accuracy"])
      # Note: "tf_keras.utils.experimental.DatasetCreator" seems to also work.
      train_dataset_creator = strategy.distribute_datasets_from_function(
          lambda context: dataset_fn(context, seed=111)
      )

      # TODO: Remove "infinite" when the valuation support finite datasets.
      valid_dataset_creator = strategy.distribute_datasets_from_function(
          lambda context: dataset_fn(context, seed=222, infinite=True)
      )
      # Note: A distributed dataset cannot be reused twice.
      valid_dataset_creator_again = strategy.distribute_datasets_from_function(
          lambda context: dataset_fn(context, seed=333, infinite=True)
      )

    # Train model
    training_history = model.fit(
        train_dataset_creator,
        validation_data=valid_dataset_creator,
        validation_steps=num_examples / batch_size,
    )
    self.assertEqual(model.num_training_examples, num_examples)

    logging.info("Training history: %s", training_history.history)
    self.assertGreaterEqual(training_history.history["val_accuracy"][0], 0.98)

    # Re-evaluate the model on a validation dataset.
    valid_evaluation_again = model.evaluate(
        valid_dataset_creator_again,
        steps=num_examples // batch_size,
        return_dict=True,
    )
    logging.info("Valid evaluation (again): %s", valid_evaluation_again)
    self.assertGreaterEqual(valid_evaluation_again["accuracy"], 0.98)

    logging.info("Trained model:")
    model.summary()

    # Check the models structure.
    inspector = model.make_inspector()
    self.assertEqual(
        [f.name for f in inspector.features()], ["data:0.0", "data:0.1"]
    )
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
      dataset = dataset.shard(
          input_context.num_input_pipelines, input_context.input_pipeline_id
      )
      dataset = dataset.batch(batch_size)
      dataset = dataset.prefetch(2)
      return dataset

    dc = keras_internal.DatasetCreator(dataset_fn)

    cluster_resolver = _create_in_process_tf_ps_cluster(num_workers=2, num_ps=1)

    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver
    )
    with strategy.scope():
      model = tfdf.keras.GradientBoostedTreesModel()

    with self.assertRaisesRegex(
        ValueError, "does not support training with a TF Distribution strategy"
    ):
      model.fit(dc, steps_per_epoch=num_examples // global_batch_size)

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

  @parameterized.named_parameters(
      ("finite_dataset_without_failures", True, False),
      ("infinite_dataset_without_failures", False, False),
      ("finite_dataset_with_failures", True, True),
  )
  def test_distributed_training_adult(
      self, use_finite_dataset, simulate_failures
  ):

    if simulate_failures:
      self.skipTest("Not tested in OSS build")

    # Split the dataset into multiple files.
    train_path = os.path.join(test_data_path(), "dataset", "adult_train.csv")
    test_path = os.path.join(test_data_path(), "dataset", "adult_test.csv")

    sharded_train_paths = self._shard_dataset(train_path)
    logging.info("Num sharded paths: %d", len(sharded_train_paths))

    batch_size = 100
    num_train_examples = pd.read_csv(train_path).shape[0]
    num_test_examples = pd.read_csv(test_path).shape[0]
    logging.info("num_train_examples: %d", num_train_examples)
    logging.info("num_test_examples: %d", num_test_examples)

    # Create the dataset
    def dataset_fn(context: distribute_lib.InputContext, paths, infinite=False):
      logging.info(
          "Create dataset with context: %s and %d path(s)", context, len(paths)
      )

      ds_path = tf.data.Dataset.from_tensor_slices(paths)

      if context is not None:
        # Split the dataset among the workers.
        current_worker = tfdf.keras.get_worker_idx_and_num_workers(context)
        assert current_worker.num_workers == 5
        ds_path = ds_path.shard(
            num_shards=current_worker.num_workers,
            index=current_worker.worker_idx,
        )

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

      column_names = [
          "age",
          "workclass",
          "fnlwgt",
          "education",
          "education_num",
          "marital_status",
          "occupation",
          "relationship",
          "race",
          "sex",
          "capital_gain",
          "capital_loss",
          "hours_per_week",
          "native_country",
          "income",
      ]
      label_name = "income"

      init_label_table = tf.lookup.KeyValueTensorInitializer(
          keys=tf.constant(["<=50K", ">50K"]),
          values=tf.constant([0, 1], dtype=tf.int64),
      )
      label_table = tf.lookup.StaticVocabularyTable(
          init_label_table, num_oov_buckets=1
      )

      def map_features(*columns):
        assert len(column_names) == len(columns)
        features = {column_names[i]: col for i, col in enumerate(columns)}
        label = label_table.lookup(features.pop(label_name))
        return features, label

      ds_dataset = ds_columns.map(map_features)
      ds_dataset = ds_dataset.batch(batch_size)
      return ds_dataset

    # Create the workers
    cluster_resolver = _create_in_process_tf_ps_cluster(num_workers=5, num_ps=1)

    # Configure the model and datasets
    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver
    )

    advanced_arguments = tfdf.keras.AdvancedArguments()
    if simulate_failures:
      gbt_training_config = advanced_arguments.yggdrasil_training_config.Extensions[
          distributed_gradient_boosted_trees_pb2.distributed_gradient_boosted_trees_config
      ]
      gbt_training_config.internal.simulate_worker_failure = True
      gbt_training_config.checkpoint_interval_trees = 5

    with strategy.scope():
      model = tfdf.keras.DistributedGradientBoostedTreesModel(
          worker_logs=False, advanced_arguments=advanced_arguments
      )
      model.compile(metrics=["accuracy"])

      train_dataset = strategy.distribute_datasets_from_function(
          lambda context: dataset_fn(
              context, sharded_train_paths, infinite=not use_finite_dataset
          )
      )

    # Train model
    model.fit(
        train_dataset,
        steps_per_epoch=None
        if use_finite_dataset
        else (num_train_examples // batch_size),
    )

    if use_finite_dataset:
      # The finite dataset approach guarenty that each example is read once and
      # exactly once. The number of training examples and dataspec statistics
      # are deterministic.

      self.assertEqual(model.num_training_examples, num_train_examples)

      inspector = model.make_inspector()
      self.assertEqual(inspector.dataspec.created_num_rows, num_train_examples)
      sex_col = inspector.dataspec.columns[13]
      self.assertEqual(sex_col.name, "sex")
      # Those figures have been computed from the raw dataset using R's table.
      self.assertEqual(sex_col.categorical.items["Female"].count, 7627)
      self.assertEqual(sex_col.categorical.items["Male"].count, 15165)

    logging.info("Trained model:")
    model.summary()

    model.save(os.path.join(tmp_path(), "pre_evaluated_model"))

    # Non-distributed evaluation of the model.
    model._distribution_strategy = None
    model._cluster_coordinator = None
    model._compile_time_distribution_strategy = None

    evaluation = model.evaluate(dataset_fn(None, [test_path]), return_dict=True)
    logging.info("Evaluation: %s", evaluation)

    model.save(os.path.join(tmp_path(), "post_evaluated_model"))

    if use_finite_dataset:
      # The finite dataset approach leads to a better model (model equivalent
      # to the non distributed training).
      self.assertAlmostEqual(evaluation["accuracy"], 0.87276, delta=0.003)
    else:
      # If all the workers are running at the same speed, at most
      # "(global_batch_size-1) * num_workers examples" can be lost before
      # training due to the Keras limitation discussed in the notes above.
      # However, because of the currently required repeat, if workers are
      # running at different speed, some examples can be repeated.
      self.assertAlmostEqual(evaluation["accuracy"], 0.8603476, delta=0.02)

  def test_distributed_training_adult_from_file(self):
    # Path to dataset.
    dataset_directory = os.path.join(test_data_path(), "dataset")
    train_path = os.path.join(dataset_directory, "adult_train.csv")
    test_path = os.path.join(dataset_directory, "adult_test.csv")

    label = "income"

    # Create the workers
    cluster_resolver = _create_in_process_tf_ps_cluster(num_workers=5, num_ps=1)

    # Configure the model and datasets
    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver
    )

    with strategy.scope():
      model = tfdf.keras.DistributedGradientBoostedTreesModel(worker_logs=False)
      model.compile(metrics=["accuracy"])

    training_history = model.fit_on_dataset_path(
        train_path=train_path,
        label_key=label,
        dataset_format="csv",
        valid_path=test_path,
    )
    logging.info("Training history: %s", training_history.history)

    logging.info("Trained model:")
    model.summary()
    _ = model.make_inspector()

    model._distribution_strategy = None
    test_df = pd.read_csv(test_path)
    tf_test = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label)
    evaluation = model.evaluate(tf_test, return_dict=True)
    logging.info("Evaluation: %s", evaluation)
    self.assertAlmostEqual(evaluation["accuracy"], 0.8703476, delta=0.01)

    features = [feature.name for feature in model.make_inspector().features()]
    self.assertEqual(
        features,
        [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education_num",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
            "native_country",
        ],
    )

  def test_distributed_hyperparameter_tuning_on_adult_from_file(self):
    # Path to dataset.
    dataset_directory = os.path.join(test_data_path(), "dataset")
    train_path = os.path.join(dataset_directory, "adult_train.csv")
    test_path = os.path.join(dataset_directory, "adult_test.csv")
    label = "income"

    # Create the workers
    cluster_resolver = _create_in_process_tf_ps_cluster(num_workers=5, num_ps=1)

    # Configure the model and datasets
    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver
    )

    with strategy.scope():
      tuner = tfdf.tuner.RandomSearch(num_trials=10, use_predefined_hps=True)
      model = tfdf.keras.GradientBoostedTreesModel(tuner=tuner, num_trees=50)
      model.compile(metrics=["accuracy"])

    training_history = model.fit_on_dataset_path(
        train_path=train_path,
        label_key=label,
        dataset_format="csv",
        valid_path=test_path,
    )
    logging.info("Training history: %s", training_history.history)

    logging.info("Trained model:")
    model.summary()
    _ = model.make_inspector()

    model._distribution_strategy = None
    test_df = pd.read_csv(test_path)
    tf_test = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label)
    evaluation = model.evaluate(tf_test, return_dict=True)
    logging.info("Evaluation: %s", evaluation)
    self.assertAlmostEqual(evaluation["accuracy"], 0.8703476, delta=0.01)

    features = [feature.name for feature in model.make_inspector().features()]
    self.assertEqual(
        features,
        [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education_num",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
            "native_country",
        ],
    )

  def test_distributed_training_adult_from_file_with_grpc_worker(self):
    self.skipTest("Not tested in OSS build")

    # Path to dataset.
    dataset_directory = os.path.join(test_data_path(), "dataset")
    train_path = os.path.join(dataset_directory, "adult_train.csv")
    test_path = os.path.join(dataset_directory, "adult_test.csv")

    label = "income"

    # Create GRPC Yggdrasil DF workers
    worker_addresses = _create_in_process_grpc_worker_cluster(5)

    # Specify the socket addresses of the worker to the manager.
    deployment_config = tfdf.keras.core.YggdrasilDeploymentConfig()
    deployment_config.try_resume_training = True
    deployment_config.distribute.implementation_key = "GRPC"
    socket_addresses = deployment_config.distribute.Extensions[
        tfdf.keras.core.grpc_pb2.grpc
    ].socket_addresses
    for worker_ip, worker_port in worker_addresses:
      socket_addresses.addresses.add(ip=worker_ip, port=worker_port)

    model = tfdf.keras.DistributedGradientBoostedTreesModel(
        worker_logs=False,
        advanced_arguments=tfdf.keras.AdvancedArguments(
            yggdrasil_deployment_config=deployment_config
        ),
    )
    model.compile(metrics=["accuracy"])

    training_history = model.fit_on_dataset_path(
        train_path=train_path,
        label_key=label,
        dataset_format="csv",
        valid_path=test_path,
    )
    logging.info("Training history: %s", training_history.history)

    logging.info("Trained model:")
    model.summary()

    test_df = pd.read_csv(test_path)
    tf_test = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label)
    evaluation = model.evaluate(tf_test, return_dict=True)
    logging.info("Evaluation: %s", evaluation)
    self.assertAlmostEqual(evaluation["accuracy"], 0.8703476, delta=0.01)

  def test_distributed_training_adult_from_file_forced_discretization(self):
    # Path to dataset.
    dataset_directory = os.path.join(test_data_path(), "dataset")
    train_path = os.path.join(dataset_directory, "adult_train.csv")
    test_path = os.path.join(dataset_directory, "adult_test.csv")

    label = "income"

    # Create the workers
    cluster_resolver = _create_in_process_tf_ps_cluster(num_workers=5, num_ps=1)

    # Configure the model and datasets
    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver
    )

    with strategy.scope():
      model = tfdf.keras.DistributedGradientBoostedTreesModel(
          worker_logs=False,
          force_numerical_discretization=True,
          max_unique_values_for_discretized_numerical=128,
      )
      model.compile(metrics=["accuracy"])

    training_history = model.fit_on_dataset_path(
        train_path=train_path,
        label_key=label,
        dataset_format="csv",
        valid_path=test_path,
    )
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
    self.assertEqual(
        features,
        [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education_num",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
            "native_country",
        ],
    )

  def test_in_memory_not_supported(self):
    dataframe = pd.DataFrame({
        "a b": [0, 1, 2],
        "c,d": [0, 1, 2],
        "e%f": [0, 1, 2],
        "a%b": [0, 1, 2],
        "label": [0, 1, 2],
    })
    dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataframe, label="label")

    model = tfdf.keras.DistributedGradientBoostedTreesModel(worker_logs=False)

    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        "does not support training from in-memory datasets",
    ):
      model.fit(dataset)


if __name__ == "__main__":
  tf.test.main()
