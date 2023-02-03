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

"""Core classes and functions of TensorFlow Decision Forests training."""

import copy
import logging
import os
import threading
import time
from typing import Any, Dict, List, NamedTuple, Optional, Union, Sequence, Tuple
import uuid

import tensorflow as tf

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute.coordinator import cluster_coordinator as cluster_coordinator_lib
from tensorflow_decision_forests.tensorflow import core_inference
from tensorflow_decision_forests.tensorflow.ops.training import op as training_op
from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.learner import abstract_learner_pb2
from yggdrasil_decision_forests.model import hyperparameter_pb2
from yggdrasil_decision_forests.utils.distribute.implementations.grpc import grpc_pb2

# Suffix added to the name of the tf resource to hold the validation
# dataset for the feature, when present. For example, if a column with id
# "ABC" has a validation dataset, its TF resources will be named "ABC" and
# "ABC__VALIDATION". See "_input_key_to_id".
_FEATURE_RESOURCE_VALIDATION_SUFFIX = "__VALIDATION"

# Imports from the inference only code.
NormalizedNumericalType = core_inference.NormalizedNumericalType
NormalizedCategoricalIntType = core_inference.NormalizedCategoricalIntType
NormalizedCategoricalStringType = core_inference.NormalizedCategoricalStringType
NormalizedCategoricalSetIntType = core_inference.NormalizedCategoricalSetIntType
NormalizedCategoricalSetStringType = (
    core_inference.NormalizedCategoricalSetStringType
)
NormalizedHashType = core_inference.NormalizedHashType
NormalizedBooleanType = core_inference.NormalizedBooleanType
SemanticTensor = core_inference.SemanticTensor
Semantic = core_inference.Semantic
Task = core_inference.Task
TaskType = core_inference.TaskType
AnyTensor = core_inference.AnyTensor
build_default_input_model_signature = (
    core_inference.build_default_input_model_signature
)
InputModelSignatureFn = core_inference.InputModelSignatureFn
build_default_feature_signature = core_inference.build_default_feature_signature
infer_semantic_from_dataframe = core_inference.infer_semantic_from_dataframe
infer_semantic = core_inference.infer_semantic
combine_tensors_and_semantics = core_inference.combine_tensors_and_semantics
normalize_inputs = core_inference.normalize_inputs
decombine_tensors_and_semantics = core_inference.decombine_tensors_and_semantics
column_type_to_semantic = core_inference.column_type_to_semantic
CATEGORICAL_INTEGER_OFFSET = core_inference.CATEGORICAL_INTEGER_OFFSET
normalize_inputs_regexp = core_inference.normalize_inputs_regexp
NodeFormat = core_inference.NodeFormat

# pylint: disable=g-import-not-at-top,import-error,unused-import,broad-except
from tensorflow.python.distribute.coordinator import coordinator_context
# pylint: enable=g-import-not-at-top,import-error,unused-import,broad-except

# A set of hyper-parameters.
# Such hyper-parameter is converted into a Yggdrasil generic hyper-parameter
# proto "GenericHyperParameters" with the function
# "hparams_dict_to_generic_proto".
HyperParameters = Dict[str, Union[int, float, str]]


class DistributionConfiguration(NamedTuple):
  """Configuration information about the distribution strategy.

  Attributes:
    num_workers: Number of workers i.e. tf worker server with the task "worker".
    workers: Network addresses of the workers.
    rpc_layer: RPC protocol.
  """

  num_workers: int
  workers: List[str]
  rpc_layer: str


def get_distribution_configuration(
    strategy: Optional[Any],
) -> Optional[DistributionConfiguration]:
  """Extracts the distribution configuration from the distribution strategy.

  Args:
    strategy: Optional distribution strategy. If none is provided, the
      distributed strategy is obtained with "tf.distribute.get_strategy".

  Returns:
    Distributed training meta-data. None if distributed training is not enabled.
  """

  if strategy is None:
    strategy = tf.distribute.get_strategy()

  # pylint:disable=protected-access
  if isinstance(
      strategy, parameter_server_strategy_v2.ParameterServerStrategyV2
  ):
    cluster_spec = strategy._cluster_resolver.cluster_spec().as_dict()
    rpc_layer = strategy._cluster_resolver.rpc_layer or "grpc"

    if not cluster_spec["worker"]:
      raise ValueError(
          "No workers configured in the distribution strategy. "
          f"Cluster spec: {cluster_spec}"
      )

    return DistributionConfiguration(
        num_workers=strategy._extended._num_workers,
        workers=cluster_spec["worker"],
        rpc_layer=rpc_layer,
    )
  elif isinstance(strategy, distribute_lib._DefaultDistributionStrategy):
    return None
  # pylint:enable=protected-access

  raise ValueError(
      f"Not supported distribution strategy {strategy}. Only "
      "no-strategy and ParameterServerStrategyV2 is supported"
  )


def get_num_distribution_workers(strategy: Optional[Any] = None) -> int:
  """Extracts the number of workers from the distribution strategy.

  Args:
    strategy: Distribution strategy to parse. If not set, use the scope
      distribution strategy (i.e. `tf.distribute.get_strategy()`)

  Returns:
    Number of workers.
  """

  distribute_config = get_distribution_configuration(strategy)
  if distribute_config is None:
    raise ValueError(
        "Number of workers not available. The method "
        "`get_num_distribution_workers` should be called within a "
        "`ParameterServerStrategyV2` scope."
    )

  return distribute_config.num_workers


class WorkerIndex(NamedTuple):
  """Index of a worker in a worker pool.

  Attributes:
    worker_idx: Index of the worker. Value in [0, num_workers).
    num_workers: Total number of workers.
  """

  worker_idx: int
  num_workers: int


def get_worker_idx_and_num_workers(
    context: distribute_lib.InputContext,
) -> WorkerIndex:
  """Gets the current worker index and the total number of workers.

  This method should be called by a worker in a tf.function called in the worker
  context. In practice, this method should be called in the in a the
  `dataset_fn(context)` method.

  Currently, `context` is ignored as it is not populated by the
  `ParameterServerStrategyV2`. However, `context` should still be provided for
  compatibility with future API changes.

  Usage examples:

    paths = [...list of dataset files]

    def dataset_fn(context: Optional[distribute_lib.InputContext] = None):
      # Distributed dataset_fn.

      ds_path = tf.data.Dataset.from_tensor_slices(paths)

      if context is not None:
        current_worker = keras.get_worker_idx_and_num_workers(context)
        assert current_worker.num_workers > 1, "Not distributed dataset reading"
        ds_path = ds_path.shard(
            num_shards=current_worker.num_workers,
            index=current_worker.worker_index)

      # Load the examples from "ds_path", for example with
      # `tf.data.experimental.CsvDataset`.

      def read_csv_file(path):
        csv_columns = [ ... ]
        return tf.data.experimental.CsvDataset(path, csv_columns, header=False)

      ds_columns = ds_path.interleave(read_csv_file)

      def extract_label(*columns):
        return columns[0:-1], columns[-1]

      return ds_columns.map(extract_label).batch(batch_size)

  Args:
    context: Distribution strategy input context.

  Returns:
    Return the index of the worker (tensor) and the total number of workers
    (integer).
  """

  if coordinator_context is None:
    raise ValueError(
        "Training with Parameter Server distributed training, however this "
        "copy of TensorFlow Decision Forests was compiled WITHOUT support for "
        "Parameter Server distributed training (TF-DF was compiled with "
        "tf_ps_distribution_strategy=0). This is a temporary but expected "
        "situation for the pre-built version of TF-DF distributed on PyPi. "
        "Your options are: (1) Don't use distributed training: Select a model "
        "that trains locally and do not use a TF Distribution Strategy. (2) "
        "Configure distributed training with GRPC distribution strategy (see "
        "details here: "
        "https://www.tensorflow.org/decision_forests/distributed_training), "
        "(3) Recompile TF on monolithic mode and TF-DF with "
        "tf_ps_distribution_strategy=1."
    )

  # Not used for now.
  del context

  if not tf.inside_function():
    raise ValueError(
        "Cannot retrieve the worker index. `get_worker_idx_and_num_workers` "
        "should be called from within a tf.function on a worker. To get the "
        "index of workers in the manager, try `context.input_pipeline_id` in "
        "a `dataset_fn(context)`."
    )

  def call_time_worker_index():
    dispatch_context = coordinator_context.get_current_dispatch_context()
    return dispatch_context.worker_index

  worker_index = tf.compat.v1.get_default_graph().capture_call_time_value(
      call_time_worker_index, tf.TensorSpec([], dtype=tf.dtypes.int64)
  )
  worker_index.op._set_attr(  # pylint: disable=protected-access
      "_user_specified_name",
      tf.compat.v1.AttrValue(s=tf.compat.as_bytes("worker_index")),
  )

  return WorkerIndex(
      worker_idx=worker_index, num_workers=get_num_distribution_workers()
  )


def column_keys_to_resource_ids(
    keys: Sequence[str], model_id: str, collect_training_data: bool
) -> Tuple[List[str], List[str]]:
  """Lists the resource id and feature names from a sequence of key.

  Args:
    keys: Sequence of keys:
    model_id: Identifier of the model.
    collect_training_data: Are those training or validation data resources.

  Returns:
    List of resource id and keys.
  """

  sorted_keys = list(keys)
  sorted(sorted_keys)

  resource_ids = []
  column_keys = []
  for key in sorted_keys:
    resource_ids.append(_input_key_to_id(model_id, key, collect_training_data))
    column_keys.append(key)
  return resource_ids, column_keys


def collect_training_examples(
    inputs: Dict[str, SemanticTensor],
    model_id: str,
    collect_training_data: Optional[bool] = True,
) -> tf.Operation:
  """Collects a batch of training examples.

  The features values are append to a set of column-wise in-memory accumulators
  contained in tf resources with respective names "_input_key_to_id(model_id,
  key)".

  Args:
    inputs: Features to collect.
    model_id: Id of the model.
    collect_training_data: Indicate if the examples are used for training.

  Returns:
    Op triggering the collection.
  """

  in_order_inputs = list(inputs.items())
  in_order_inputs.sort(key=lambda x: x[0])

  ops = []
  for key, semantic_tensor in in_order_inputs:

    def raise_non_supported():
      raise Exception(
          "Non supported tensor dtype {} and semantic {} for feature {}".format(
              semantic_tensor.tensor.dtype, semantic_tensor.semantic, key
          )
      )  # pylint: disable=cell-var-from-loop

    input_id = _input_key_to_id(model_id, key, collect_training_data)

    if semantic_tensor.semantic in [
        Semantic.NUMERICAL,
        Semantic.DISCRETIZED_NUMERICAL,
    ]:
      if semantic_tensor.tensor.dtype == NormalizedNumericalType:
        ops.append(
            training_op.simple_ml_numerical_feature(
                value=semantic_tensor.tensor, id=input_id, feature_name=key
            )
        )
      else:
        raise_non_supported()

    elif semantic_tensor.semantic == Semantic.CATEGORICAL:
      if semantic_tensor.tensor.dtype == NormalizedCategoricalStringType:
        ops.append(
            training_op.simple_ml_categorical_string_feature(
                value=semantic_tensor.tensor, id=input_id, feature_name=key
            )
        )
      elif semantic_tensor.tensor.dtype == NormalizedCategoricalIntType:
        ops.append(
            training_op.simple_ml_categorical_int_feature(
                value=semantic_tensor.tensor, id=input_id, feature_name=key
            )
        )
      else:
        raise_non_supported()

    elif semantic_tensor.semantic == Semantic.CATEGORICAL_SET:
      args = {
          "values": semantic_tensor.tensor.values,
          "row_splits": semantic_tensor.tensor.row_splits,
          "id": input_id,
          "feature_name": key,
      }
      if semantic_tensor.tensor.dtype == NormalizedCategoricalSetStringType:
        ops.append(training_op.simple_ml_categorical_set_string_feature(**args))
      elif semantic_tensor.tensor.dtype == NormalizedCategoricalIntType:
        ops.append(training_op.simple_ml_categorical_set_int_feature(**args))
      else:
        raise_non_supported()

    elif semantic_tensor.semantic == Semantic.HASH:
      if semantic_tensor.tensor.dtype == NormalizedHashType:
        ops.append(
            training_op.simple_ml_hash_feature(
                value=semantic_tensor.tensor, id=input_id, feature_name=key
            )
        )
      else:
        raise_non_supported()

    elif semantic_tensor.semantic == Semantic.BOOLEAN:
      # Boolean features are not yet supported for training in TF-DF.
      raise_non_supported()

    else:
      raise_non_supported()

  return tf.group(ops)


def collect_distributed_training_examples(
    inputs: Dict[str, SemanticTensor], model_id: str, dataset_path: str
) -> tf.Operation:
  """Exports feature values to file in the partial dataset cache format.

  For distributed training, multiple tasks (with task="worker" and different
  task index) can collect feature values at the same time and in the same
  `dataset_path` location.

  Once feature values are done being collected,
  "finalize_distributed_dataset_collection" should be called.

  Args:
    inputs: Feature values to collect.
    model_id: Id of the model.
    dataset_path: Directory path to the output partial dataset cache.

  Returns:
    Op triggering the collection.
  """

  in_order_inputs = list(inputs.items())
  in_order_inputs.sort(key=lambda x: x[0])

  ops = []
  for feature_idx, (feature_name, semantic_tensor) in enumerate(
      in_order_inputs
  ):
    def raise_non_supported():
      # pylint: disable=cell-var-from-loop
      raise Exception(
          f"Non supported tensor dtype {semantic_tensor.tensor.dtype} "
          f"and semantic {semantic_tensor.semantic} for feature {feature_name} "
          "for distributed training"
      )
      # pylint: enable=cell-var-from-loop

    resource_id = _input_key_to_id(model_id, feature_name, training_column=True)
    if semantic_tensor.semantic == Semantic.NUMERICAL:
      if semantic_tensor.tensor.dtype == NormalizedNumericalType:
        ops.append(
            training_op.SimpleMLNumericalFeatureOnFile(
                value=semantic_tensor.tensor,
                resource_id=resource_id,
                feature_name=feature_name,
                feature_idx=feature_idx,
                dataset_path=dataset_path,
            )
        )
      else:
        raise_non_supported()

    elif semantic_tensor.semantic == Semantic.CATEGORICAL:
      if semantic_tensor.tensor.dtype == NormalizedCategoricalIntType:
        ops.append(
            training_op.SimpleMLCategoricalIntFeatureOnFile(
                value=semantic_tensor.tensor,
                resource_id=resource_id,
                feature_name=feature_name,
                feature_idx=feature_idx,
                dataset_path=dataset_path,
            )
        )
      elif semantic_tensor.tensor.dtype == NormalizedCategoricalStringType:
        ops.append(
            training_op.SimpleMLCategoricalStringFeatureOnFile(
                value=semantic_tensor.tensor,
                resource_id=resource_id,
                feature_name=feature_name,
                feature_idx=feature_idx,
                dataset_path=dataset_path,
            )
        )
      else:
        raise_non_supported()

    else:
      raise_non_supported()

  return tf.group(ops)


def check_config(
    generic_hparms: hyperparameter_pb2.GenericHyperParameters,
    training_config: abstract_learner_pb2.TrainingConfig,
):
  """Checks the validity of a training configuration."""

  try:
    training_op.SimpleMLCheckTrainingConfiguration(
        hparams=generic_hparms.SerializeToString(),
        training_config=training_config.SerializeToString(),
    )
  except tf.errors.UnknownError as e:
    raise ValueError(e.message)


def train(
    resource_ids: List[str],
    model_id: str,
    generic_hparms: Optional[hyperparameter_pb2.GenericHyperParameters] = None,
    training_config: Optional[abstract_learner_pb2.TrainingConfig] = None,
    deployment_config: Optional[abstract_learner_pb2.DeploymentConfig] = None,
    guide: Optional[data_spec_pb2.DataSpecificationGuide] = None,
    model_dir: Optional[str] = None,
    keep_model_in_resource: Optional[bool] = True,
    try_resume_training: Optional[bool] = False,
    has_validation_dataset: Optional[bool] = False,
    node_format: Optional[NodeFormat] = None,
):
  """Trains a model on the dataset accumulated by collect_training_examples.

  Args:
    resource_ids: Id of the tf resources containing the feature values.
    model_id: Id of the model.
    generic_hparms: Hyper-parameter of the learner.
    training_config: Training configuration.
    deployment_config: Deployment configuration (e.g. where to train the model).
    guide: Dataset specification guide.
    model_dir: If specified, export the trained model into this directory.
    keep_model_in_resource: If true, keep the model as a training model
      resource.
    try_resume_training: Try to resume the training from the
      "working_cache_path" directory. The the "working_cache_path" does not
      contains any checkpoint, start the training from the start.
    has_validation_dataset: True if a validation dataset is available (in
      addition to the training dataset).
    node_format: Format for storing a model's nodes used by Yggdrasil Decision
      Forests.

  Returns:
    The OP that trigger the training.
  """

  if generic_hparms is None:
    generic_hparms = hyperparameter_pb2.GenericHyperParameters()

  if training_config is None:
    training_config = abstract_learner_pb2.TrainingConfig()
  else:
    training_config = copy.deepcopy(training_config)

  if deployment_config is None:
    deployment_config = abstract_learner_pb2.DeploymentConfig()
  else:
    deployment_config = copy.deepcopy(deployment_config)

  if try_resume_training:
    deployment_config.cache_path = os.path.join(model_dir, "working_cache")
    deployment_config.try_resume_training = True

  if guide is None:
    guide = data_spec_pb2.DataSpecificationGuide()
  else:
    guide = copy.deepcopy(guide)

  process_id = training_op.SimpleMLModelTrainer(
      resource_ids=resource_ids,
      model_id=model_id,
      model_dir=model_dir or "",
      hparams=generic_hparms.SerializeToString(),
      training_config=training_config.SerializeToString(),
      deployment_config=deployment_config.SerializeToString(),
      guide=guide.SerializeToString(),
      has_validation_dataset=has_validation_dataset,
      use_file_prefix=True,
      create_model_resource=keep_model_in_resource,
      node_format="" if node_format is None else node_format,
  )

  if process_id != -1:
    # Wait for the training to be done.
    while True:
      if (
          training_op.SimpleMLCheckStatus(process_id=process_id) == 1
      ):  # kSuccess
        break


def train_on_file_dataset(
    train_dataset_path: str,
    valid_dataset_path: Optional[str],
    model_id: str,
    generic_hparms: Optional[hyperparameter_pb2.GenericHyperParameters] = None,
    training_config: Optional[abstract_learner_pb2.TrainingConfig] = None,
    deployment_config: Optional[abstract_learner_pb2.DeploymentConfig] = None,
    guide: Optional[data_spec_pb2.DataSpecificationGuide] = None,
    model_dir: Optional[str] = None,
    keep_model_in_resource: Optional[bool] = True,
    working_cache_path: Optional[str] = None,
    distribution_config: Optional[DistributionConfiguration] = None,
    try_resume_training: Optional[bool] = False,
    cluster_coordinator: Optional[Any] = None,
    node_format: Optional[NodeFormat] = None,
    force_ydf_port: Optional[int] = None,
):
  """Trains a model on dataset stored on file.

  The input arguments and overall logic of this OP is similar to the ":train"
  CLI or the "learner->Train()" method of Yggdrasil Decision Forests (in fact,
  this OP simply calls "learner->Train()").

  Similarly as the `train` method, the implementation the learning algorithm
  should be added as a dependency to the binary. Similarly, the implementation
  the dataset format should be added as a dependency to the
  binary.

  In the case of distributed training, `train_on_file_dataset` should only be
  called by the `chief` process, and `deployment_config` should contain the
  address of the workers.

  Args:
    train_dataset_path: Path to the training dataset.
    valid_dataset_path: Path to the validation dataset.
    model_id: Id of the model.
    generic_hparms: Hyper-parameter of the learner.
    training_config: Training configuration.
    deployment_config: Deployment configuration (e.g. where to train the model).
    guide: Dataset specification guide.
    model_dir: If specified, export the trained model into this directory.
    keep_model_in_resource: If true, keep the model as a training model
      resource.
    working_cache_path: Path to the working cache directory. If set, and if the
      training is distributed, all the workers should have write access to this
      cache.
    distribution_config: Socket addresses of the workers for distributed
      training.
    try_resume_training: Try to resume the training from the
      "working_cache_path" directory. The the "working_cache_path" does not
      contains any checkpoint, start the training from the start.
    cluster_coordinator: Cluster coordinator of the distributed training.
    node_format: Format for storing a model's nodes used by Yggdrasil Decision
      Forests.
    force_ydf_port: Port for YDF to use. The chief and the workers should be
      able to communicate thought this port. If not set, an available port is
      automatically selected.

  Returns:
    The OP that trigger the training.
  """

  if generic_hparms is None:
    generic_hparms = hyperparameter_pb2.GenericHyperParameters()

  if training_config is None:
    training_config = abstract_learner_pb2.TrainingConfig()
  else:
    training_config = copy.deepcopy(training_config)

  if deployment_config is None:
    deployment_config = abstract_learner_pb2.DeploymentConfig()
  else:
    deployment_config = copy.deepcopy(deployment_config)

  if guide is None:
    guide = data_spec_pb2.DataSpecificationGuide()
  else:
    guide = copy.deepcopy(guide)

  if working_cache_path is not None:
    deployment_config.cache_path = working_cache_path

  if try_resume_training:
    if working_cache_path is None:
      raise ValueError(
          "Cannot train a model with `try_resume_training=True` "
          "without a working cache directory."
      )
    deployment_config.try_resume_training = True

  # Thread in charge with checking the status of workers.
  check_workers_thread = None

  # Indicates when to stop the "check_workers_thread" thread.
  check_worker_stop = False

  # Key to identify the GRPC session in the chief and worker processes.
  grpc_session_key = None

  if distribution_config is not None:
    logging.info("Start GRPC workers")

    # Configure the GRPC YDF workers inside of TF workers.
    assert cluster_coordinator is not None
    if distribution_config.workers is None:
      raise ValueError("No workers configured")

    # The key is stored as a int32.
    grpc_session_key = uuid.uuid4().int & 0x7FFFFFFF

    # Start the GRPC workers.
    #
    # Note: The GRPC addresses (used by the GRPC worker) are different from the
    # TF addresses used by the TF workers.
    tf_workers_addresses = distribution_config.workers
    grpc_workers_addresses = ensure_grpc_workers_are_running(
        cluster_coordinator=cluster_coordinator,
        grpc_session_key=grpc_session_key,
        tf_workers_addresses=tf_workers_addresses,
        force_ydf_port=force_ydf_port,
    )

    deployment_config.try_resume_training = True
    deployment_config.distribute.implementation_key = "GRPC"
    grpc_dist_config = deployment_config.distribute.Extensions[grpc_pb2.grpc]
    grpc_dist_config.key = grpc_session_key
    grpc_dist_config.grpc_addresses.addresses[:] = grpc_workers_addresses

    def check_workers(
        grpc_workers_addresses: List[str], tf_workers_addresses: List[str]
    ) -> None:
      """Continuously checks the status of GRPC workers.

      This function is responsible for restarting GRPC workers and retrieving
      the new port in case of worker preemption.

      Args:
        grpc_workers_addresses: Initial list of addresses of the workers.
        tf_workers_addresses: Address of the TF workers.
      """

      while not check_worker_stop:
        # Run the check every 30 seconds.
        time.sleep(30)

        new_grpc_workers_addresses = ensure_grpc_workers_are_running(
            cluster_coordinator=cluster_coordinator,
            grpc_session_key=grpc_session_key,
            tf_workers_addresses=tf_workers_addresses,
            force_ydf_port=force_ydf_port,
        )

        assert len(new_grpc_workers_addresses) == len(grpc_workers_addresses)
        for worker_idx, grpc_workers_address in enumerate(
            grpc_workers_addresses
        ):
          if new_grpc_workers_addresses[worker_idx] != grpc_workers_address:
            logging.info(
                "Update worker #%d port from %d to %d",
                worker_idx,
                grpc_workers_address,
                new_grpc_workers_addresses[worker_idx],
            )
            training_op.SimpleMLUpdateGRPCWorkerAddress(
                key=grpc_session_key,
                worker_idx=worker_idx,
                new_address=new_grpc_workers_addresses[worker_idx],
            )

        grpc_workers_addresses = new_grpc_workers_addresses

    check_workers_thread = threading.Thread(
        target=check_workers,
        args=(
            grpc_workers_addresses,
            tf_workers_addresses,
        ),
        daemon=True,
    )
    check_workers_thread.start()

  # Start the chief training logic.
  process_id = training_op.SimpleMLModelTrainerOnFile(
      train_dataset_path=train_dataset_path,
      valid_dataset_path=valid_dataset_path if valid_dataset_path else "",
      model_id=model_id,
      model_dir=model_dir or "",
      hparams=generic_hparms.SerializeToString(),
      training_config=training_config.SerializeToString(),
      deployment_config=deployment_config.SerializeToString(),
      guide=guide.SerializeToString(),
      use_file_prefix=True,
      create_model_resource=keep_model_in_resource,
      node_format="" if node_format is None else node_format,
  )

  if process_id != -1:
    # Wait for the chief training logic to be done.
    #
    # Note: The chief is responsible to stop the GRPC workers. However, in case
    # of preemption of the worker during the shut-down phase, it is possible
    # for a GRPC worker to remain active.
    while True:
      if (
          training_op.SimpleMLCheckStatus(process_id=process_id) == 1
      ):  # kSuccess
        break

  if distribution_config is not None:
    # Stop the grpc worker checker
    check_worker_stop = True
    if check_workers_thread is not None:
      check_workers_thread.join()

    # Stop the grpc workers / make sure they are stopped.
    stop_grpc_workers(cluster_coordinator, grpc_session_key)


def finalize_distributed_dataset_collection(
    cluster_coordinator,
    feature_names: List[str],
    resource_ids: List[str],
    dataset_path: str,
) -> None:
  """Finaliazes the collection of the partial dataset cache.

  Args:
    cluster_coordinator: Cluster coordinator of the distributed training.
    feature_names: Key of the features/columns to collect.
    resource_ids: TF resource id for the columns to collect.
    dataset_path: Directory path to the output partial dataset cache.
  """

  def worker_fn():
    training_op.SimpleMLWorkerFinalizeFeatureOnFile(
        feature_resource_ids=resource_ids, dataset_path=dataset_path
    )

  execute_function_on_each_worker(cluster_coordinator, worker_fn)

  training_op.SimpleMLChiefFinalizeFeatureOnFile(
      feature_names=feature_names,
      num_shards=len(cluster_coordinator._cluster.workers),  # pylint: disable=protected-access
      dataset_path=dataset_path,
  )


def execute_function_on_each_worker(
    coordinator: Any,
    call_fn: Any,
    args: Any = None,
    reduce_results: bool = True,
) -> Union[Any, List[Any]]:
  """Blocking execution of `call_fn` once on each of the workers in parallel.

  Unlike "execute_function_on_each_worker" that use directly the "device" API,
  this function uses the closure API of the coordinator: The call_fn is
  automatically with coordinator data, and args can be a PerWorker iterator.

  Args:
    coordinator: PSStrategy coordinate.
    call_fn: Function to run remotely.
    args: Arguments of call_fn. If args contains PerWorkers arguments, each
      worker will only receive the arguments for them.
    reduce_results: If true, reduces the results with the sum (+) operator. If
      false, returns the individual results sorted by worker index.

  Returns:
    The sum (+) of the call_fn return values (if reduce_results=True), or
    individual results (if
    reduce_results=False).
  """
  # pylint: disable=protected-access

  args = args or ()

  class Result(object):
    """Mutable structure containing the accumulated data."""

    def __init__(self):
      self.worker_idxs = []
      if reduce_results:
        self.value = None
      else:
        self.value = []
      self.lock = threading.Lock()

    def add(self, value, worker_idx):
      """Add a value."""

      if value is None:
        return
      self.lock.acquire()
      self.worker_idxs.append(worker_idx)
      if reduce_results:
        if self.value is None:
          self.value = value
        else:
          self.value += value
      else:
        self.value.append(value)
      self.lock.release()

  result = Result()

  def thread_body(worker_idx, result):
    closure = cluster_coordinator_lib.Closure(
        call_fn, coordinator._cluster.closure_queue._cancellation_mgr, args=args
    )
    ret = closure.build_output_remote_value()

    def run_my_closure():
      closure.execute_on(coordinator._cluster.workers[worker_idx])

    with coordinator._cluster.failure_handler.wait_on_failure(
        on_recovery_fn=run_my_closure, worker_device_name=f"worker {worker_idx}"
    ):
      run_my_closure()

    ret_value = ret.get()
    if ret_value is not None:
      result.add(ret_value.numpy(), worker_idx)

  threads = []
  for worker_idx in range(coordinator._strategy._extended._num_workers):
    thread = threading.Thread(
        target=thread_body,
        args=(
            worker_idx,
            result,
        ),
        daemon=True,
    )
    thread.start()
    threads.append(thread)

  for thread in threads:
    thread.join()

  if reduce_results:
    return result.value
  else:
    values = list(zip(result.worker_idxs, result.value))
    values.sort()
    return [x[1] for x in values]
  # pylint: enable=protected-access


def ensure_grpc_workers_are_running(
    cluster_coordinator,
    grpc_session_key: int,
    tf_workers_addresses: List[str],
    force_ydf_port: Optional[int],
) -> List[str]:
  """Ensures that a GRPC YDF worker is running on each of the TF Workers.

  If a TF Worker is not running a GRPC YDF worker, create the missing GRPC YDF
  worker.

  Args:
    cluster_coordinator: A TF cluster coordinate.
    grpc_session_key: Identifier of the GRPC session.
    tf_workers_addresses: Address of the TF workers.
    force_ydf_port: Port for YDF to use. The chief and the workers should be
      able to communicate thought this port. If not set, an available port is
      automatically selected.

  Returns:
    The addresses of the workers.
  """

  def worker_fn():
    return training_op.SimpleMLCreateYDFGRPCWorker(
        key=grpc_session_key, force_ydf_port=force_ydf_port
    )

  grpc_ports = execute_function_on_each_worker(
      cluster_coordinator, worker_fn, reduce_results=False
  )

  assert len(tf_workers_addresses) == len(grpc_ports)

  addresses = []
  for worker_idx in range(len(tf_workers_addresses)):
    tf_workers_address = tf_workers_addresses[worker_idx]
    addresses.append(
        replace_port_in_address(tf_workers_address, grpc_ports[worker_idx])
    )
  return addresses


def stop_grpc_workers(cluster_coordinator, grpc_session_key: int) -> None:
  """Stops all GRPC YDF workers.

  Args:
    cluster_coordinator: A TF cluster coordinate.
    grpc_session_key: Identifier of the GRPC session.
  """

  def worker_fn():
    return training_op.SimpleMLStopYDFGRPCWorker(key=grpc_session_key)

  execute_function_on_each_worker(
      cluster_coordinator, worker_fn, reduce_results=False
  )


def replace_port_in_address(address: str, new_port: int) -> str:
  """Replaces the socket port in an address.

  Args:
    address: Source address e.g., "127.0.0.1:1234"
    new_port: New port.

  Returns:
    The address with the changed port.
  """

  address_parts = address.rsplit(":", maxsplit=1)

  if len(address_parts) == 2:
    # The address is in a [address]:[port] format.
    return f"{address_parts[0]}:{new_port}"

  else:
    raise ValueError(f"Cannot parse worker address: {address}")


def _input_key_to_id(model_id: str, key: str, training_column: bool) -> str:
  """Gets the name of the feature accumulator resource."""

  # Escape the commas that are used to separate the column resource id.
  # Those IDs have not impact to the final model, but they should be unique and
  # not contain commas.
  #
  # Turn the character '|' into an escape symbol.
  input_id = model_id + "_" + key.replace("|", "||").replace(",", "|c")
  if "," in input_id:
    raise ValueError(f"Internal error: Found comma in input_id {input_id}")
  if not training_column:
    input_id += _FEATURE_RESOURCE_VALIDATION_SUFFIX
  return input_id


def hparams_dict_to_generic_proto(
    hparams: Optional[HyperParameters] = None,
) -> hyperparameter_pb2.GenericHyperParameters:
  """Converts hyper-parameters from dict to proto representation."""

  generic = hyperparameter_pb2.GenericHyperParameters()
  if hparams is None:
    return generic

  for key, value in hparams.items():
    if value is None:
      continue
    field = generic.fields.add()
    field.name = key
    if isinstance(value, bool):
      field.value.categorical = "true" if value else "false"
    elif isinstance(value, int):
      field.value.integer = value
    elif isinstance(value, float):
      field.value.real = value
    elif isinstance(value, str):
      field.value.categorical = value
    else:
      raise Exception(
          'Unsupported type "{}:{}" for hyper-parameter "{}". '
          "Possible types are int (for integer), float (for real), and str "
          "(for categorical)".format(value, type(value), key)
      )

  return generic
