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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import enum
import logging
import math
import re
from typing import Optional, List, Tuple, Dict, Union, NamedTuple, Callable, Text, Any

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow_decision_forests.tensorflow.distribute import tf_distribution_pb2
from tensorflow_decision_forests.tensorflow.ops.training import api as training_op
from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.learner import abstract_learner_pb2
from yggdrasil_decision_forests.model import abstract_model_pb2

try:
  from tensorflow_decision_forests.tensorflow.distribute import api as distributed_api  # pytype: disable=import-error
  from tensorflow.python.distribute.coordinator import coordinator_context  # pytype: disable=import-error
except Exception as e:
  distributed_api = None
  coordinator_context = None
  logging.warning("TF Parameter Server distributed training not available.")


class Semantic(enum.Enum):
  """Semantic (e.g.

  numerical, categorical) of an input feature.

  Determines how a feature is interpreted by the model.
  Similar to the "column type" of Yggdrasil Decision Forest.

  Attributes:
    NUMERICAL: Numerical value. Generally for quantities or counts with full
      ordering. For example, the age of a person, or the number of items in a
      bag. Can be a float or an integer.  Missing values are represented by
      math.nan or with an empty sparse tensor.  If a numerical tensor contains
      multiple values, its size should be constant, and each dimension is
      threaded independently (and each dimension should always have the same
      "meaning").
    CATEGORICAL: A categorical value. Generally for a type/class in finite set
      of possible values without ordering. For example, the color RED in the set
      {RED, BLUE, GREEN}. Can be a string or an integer.  Missing values are
      represented by "" (empty sting), value -2 or with an empty sparse tensor.
      An out-of-vocabulary value (i.e. a value that was never seen in training)
      is represented by any new string value or the value -1.  If a numerical
      tensor contains multiple values, its size should be constant, and each
      value is treated independently (each value on the tensor should always
      have the same meaning).
      Integer categorical values: (1) The training logic and model
        representation is optimized with the assumption that values are dense.
        (2) Internally, the value is stored as int32. The values should be <~2B.
        (3) The number of possible value is computed automatically from the
        training dataset. During inference, integer values greater than any
        value seen during training will be treated as out-of-vocabulary. (4)
        Minimum frequency and maximum vocabulary size constrains don't apply.
    HASH: The hash of a string value. Used when only the equality between values
      is important (not the value itself). Currently, only used for groups in
      ranking problems e.g. the query in a query/document problem. The hashing
      is computed with google's farmhash and stored as an uint64.
    CATEGORICAL_SET: Set of categorical values. Great to represent tokenized
      texts. Can be a string or an integer in a sparse tensor or a ragged tensor
      (recommended). Unlike CATEGORICAL, the number of items in a
      CATEGORICAL_SET can change and the order/index of each item doesn't
      matter.
    BOOLEAN: Boolean value. WARNING: Boolean values are not yet supported for
      training. Can be a float or an integer. Missing values are represented by
      math.nan or with an empty sparse tensor.  If a numerical tensor contains
      multiple values, its size should be constant, and each dimension is
      threaded independently (and each dimension should always have the same
      "meaning").
  """

  NUMERICAL = 1
  CATEGORICAL = 2
  HASH = 3
  CATEGORICAL_SET = 4
  BOOLEAN = 5


# Any tensorflow tensor.
AnyTensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]


class SemanticTensor(NamedTuple):
  """Combination of a tensor and its semantic."""
  semantic: Semantic
  tensor: AnyTensor = None


# The available tensor dtypes for each semantic.
FlexibleNumericalTypes = [
    tf.float32, tf.int8, tf.int16, tf.int32, tf.int64, tf.float64
]
FlexibleCategoricalIntTypes = [tf.int8, tf.int16, tf.int32, tf.int64]
FlexibleCategoricalStringTypes = [tf.string]
FlexibleCategoricalSetIntTypes = FlexibleCategoricalIntTypes
FlexibleCategoricalSetStringTypes = FlexibleCategoricalStringTypes
FlexibleHashTypes = [tf.string]
FlexibleBooleanTypes = FlexibleNumericalTypes

# The "normalized" dtype for each semantic.
# Those are the dtypes expected by the tf ops.
# By definition, the normalized dtype of each semantic is part of the flexible
# dtypes.
NormalizedNumericalType = tf.float32
NormalizedCategoricalIntType = tf.int32
NormalizedCategoricalStringType = tf.string
NormalizedCategoricalSetIntType = NormalizedCategoricalIntType
NormalizedCategoricalSetStringType = NormalizedCategoricalStringType
NormalizedHashType = tf.string
NormalizedBooleanType = NormalizedNumericalType

# Magic offset to add to categorical feature stored as integer.
# Yggdrasil reserves the integer value "0" of CATEGORICAL features for the
# "out of vocabulary" item. Since it is common to have integer categorical
# feature values using 0 as a normal value, Yggdrasil sees always the original
# value +1.
# Note: Negative integer value for categorical features are forbidden.
CATEGORICAL_INTEGER_OFFSET = 1

# A set of hyper-parameters.
# Such hyper-parameter is converted into a Yggdrasil generic hyper-parameter
# proto "GenericHyperParameters" with the function
# "hparams_dict_to_generic_proto".
HyperParameters = Dict[str, Union[int, float, str]]

# The task of a model e.g. classification, regression, ranking.
Task = abstract_model_pb2.Task
TaskType = "abstract_model_pb2.Task"  # pylint: disable=invalid-name


class DistributionConfiguration(NamedTuple):
  """Configuration information about the distribution strategy.

  Attributes:
    num_workers: Number of workers i.e. tf worker server with the task "worker".
    workers_addresses: Network address of the workers e.g. grpc://127.0.0.1:1234
  """

  num_workers: int
  workers_addresses: Optional[List[str]]


def get_distribution_configuration(
    strategy: Optional[Any]) -> Optional[DistributionConfiguration]:
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
  if isinstance(strategy,
                parameter_server_strategy_v2.ParameterServerStrategyV2):

    cluster_spec = strategy._cluster_resolver.cluster_spec().as_dict()
    rpc_layer = strategy._cluster_resolver.rpc_layer or "grpc"
    workers_addresses = []
    for worker_address in cluster_spec["worker"]:
      workers_addresses.append(f"{rpc_layer}://{worker_address}")

    if not workers_addresses:
      logging.warning(
          "Empty worker network addresses in the distribution strategy. "
          "Using TF_CONFIG instead. Cluster spec: %s", cluster_spec)
      workers_addresses = None

    return DistributionConfiguration(
        num_workers=strategy._extended._num_workers,
        workers_addresses=workers_addresses)
  elif isinstance(strategy, distribute_lib._DefaultDistributionStrategy):
    return None
  # pylint:enable=protected-access

  raise ValueError(f"Not supported distribution strategy {strategy}. Only "
                   "no-strategy and ParameterServerStrategyV2 is supported")


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
    raise ValueError("Number of workers not available. The method "
                     "`get_num_distribution_workers` should be called within a "
                     "`ParameterServerStrategyV2` scope.")

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
    context: distribute_lib.InputContext) -> WorkerIndex:
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
        "The library was compile without Parameter Server distributed training "
        " support i.e. tf_ps_distribution_strategy=0. Either recompile it with "
        "tf_ps_distribution_strategy=1,disable distributed training, or use "
        "the Grpc Server distribution strategy. Note: TF-DF OSS release it "
        "currently compiled without PS support.")

  # Not used for now.
  del context

  if not tf.inside_function():
    raise ValueError(
        "Cannot retrieve the worker index. `get_worker_idx_and_num_workers` "
        "should be called from within a tf.function on a worker. To get the "
        "index of workers in the manager, try `context.input_pipeline_id` in "
        "a `dataset_fn(context)`.")

  def call_time_worker_index():
    dispatch_context = coordinator_context.get_current_dispatch_context()
    return dispatch_context.worker_index

  worker_index = tf.compat.v1.get_default_graph().capture_call_time_value(
      call_time_worker_index, tf.TensorSpec([], dtype=tf.dtypes.int64))
  worker_index.op._set_attr(  # pylint: disable=protected-access
      "_user_specified_name",
      tf.compat.v1.AttrValue(s=tf.compat.as_bytes("worker_index")))

  return WorkerIndex(
      worker_idx=worker_index, num_workers=get_num_distribution_workers())


def collect_training_examples(inputs: Dict[str, SemanticTensor],
                              model_id: str) -> tf.Operation:
  """Collects a batch of training examples.

  The features values are append to a set of column-wise in-memory accumulators
  contained in tf resources with respective names "_input_key_to_id(model_id,
  key)".

  Args:
    inputs: Features to collect.
    model_id: Id of the model.

  Returns:
    Op triggering the collection.
  """

  ops = []
  for key, semantic_tensor in inputs.items():

    def raise_non_supported():
      raise Exception(
          "Non supported tensor dtype {} and semantic {} for feature {}".format(
              semantic_tensor.tensor.dtype, semantic_tensor.semantic, key))  # pylint: disable=cell-var-from-loop

    input_id = _input_key_to_id(model_id, key)
    if semantic_tensor.semantic == Semantic.NUMERICAL:
      if semantic_tensor.tensor.dtype == NormalizedNumericalType:
        ops.append(
            training_op.simple_ml_numerical_feature(
                value=semantic_tensor.tensor, id=input_id, feature_name=key))
      else:
        raise_non_supported()

    elif semantic_tensor.semantic == Semantic.CATEGORICAL:
      if semantic_tensor.tensor.dtype == NormalizedCategoricalStringType:
        ops.append(
            training_op.simple_ml_categorical_string_feature(
                value=semantic_tensor.tensor, id=input_id, feature_name=key))
      elif semantic_tensor.tensor.dtype == NormalizedCategoricalIntType:
        ops.append(
            training_op.simple_ml_categorical_int_feature(
                value=semantic_tensor.tensor, id=input_id, feature_name=key))
      else:
        raise_non_supported()

    elif semantic_tensor.semantic == Semantic.CATEGORICAL_SET:
      args = {
          "values": semantic_tensor.tensor.values,
          "row_splits": semantic_tensor.tensor.row_splits,
          "id": input_id,
          "feature_name": key
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
                value=semantic_tensor.tensor, id=input_id, feature_name=key))
      else:
        raise_non_supported()

    elif semantic_tensor.semantic == Semantic.BOOLEAN:
      # Boolean features are not yet supported for training in TF-DF.
      raise_non_supported()

    else:
      raise_non_supported()

  return tf.group(ops)


def collect_distributed_training_examples(inputs: Dict[str, SemanticTensor],
                                          model_id: str,
                                          dataset_path: str) -> tf.Operation:
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
  for feature_idx, (feature_name,
                    semantic_tensor) in enumerate(in_order_inputs):

    def raise_non_supported():
      # pylint: disable=cell-var-from-loop
      raise Exception(
          f"Non supported tensor dtype {semantic_tensor.tensor.dtype} "
          f"and semantic {semantic_tensor.semantic} for feature {feature_name} for "
          "distributed  training")
      # pylint: enable=cell-var-from-loop

    resource_id = _input_key_to_id(model_id, feature_name)
    if semantic_tensor.semantic == Semantic.NUMERICAL:
      if semantic_tensor.tensor.dtype == NormalizedNumericalType:
        ops.append(
            training_op.SimpleMLNumericalFeatureOnFile(
                value=semantic_tensor.tensor,
                resource_id=resource_id,
                feature_name=feature_name,
                feature_idx=feature_idx,
                dataset_path=dataset_path))
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
                dataset_path=dataset_path))
      elif semantic_tensor.tensor.dtype == NormalizedCategoricalStringType:
        ops.append(
            training_op.SimpleMLCategoricalStringFeatureOnFile(
                value=semantic_tensor.tensor,
                resource_id=resource_id,
                feature_name=feature_name,
                feature_idx=feature_idx,
                dataset_path=dataset_path))
      else:
        raise_non_supported()

    else:
      raise_non_supported()

  return tf.group(ops)


def normalize_inputs(
    inputs: Dict[str, SemanticTensor]) -> Dict[str, SemanticTensor]:
  """Normalize input tensors for OP consumption.

  Normalization involves:
    - Casting the feature to the "normalized" dtype.
    - Dividing fixed-length features into sets of single dimensional features.
    - Converts the sparse tensors (generally used to represent possible missing
      values) into dense tensors with the correct missing value representation.

  Args:
    inputs: Dict of semantic tensor.

  Returns:
    Dict of normalized semantic tensors.

  Raises:
    ValueError: The arguments are invalid.
  """

  normalized_inputs = {}

  for key, semantic_tensor in inputs.items():
    if semantic_tensor.semantic == Semantic.NUMERICAL:
      if semantic_tensor.tensor.dtype in FlexibleNumericalTypes:
        _unroll_and_normalize(
            tf.cast(semantic_tensor.tensor, tf.float32),
            semantic_tensor.semantic, key, math.nan, normalized_inputs)
      else:
        raise ValueError(
            "Non supported tensor dtype {} for semantic {} of feature {}"
            .format(semantic_tensor.tensor.dtype, semantic_tensor.semantic,
                    key))

    elif semantic_tensor.semantic == Semantic.CATEGORICAL:
      if semantic_tensor.tensor.dtype in FlexibleCategoricalStringTypes:
        _unroll_and_normalize(
            tf.cast(semantic_tensor.tensor, tf.string),
            semantic_tensor.semantic, key, "", normalized_inputs)
      elif semantic_tensor.tensor.dtype in FlexibleCategoricalIntTypes:
        _unroll_and_normalize(
            tf.cast(semantic_tensor.tensor, tf.int32),
            semantic_tensor.semantic,
            key,
            -1 - CATEGORICAL_INTEGER_OFFSET,
            normalized_inputs,
            dense_preprocess=lambda x: x + CATEGORICAL_INTEGER_OFFSET)
      else:
        raise ValueError(
            "Non supported tensor dtype {} for semantic {} of feature {}"
            .format(semantic_tensor.tensor.dtype, semantic_tensor.semantic,
                    key))

    elif semantic_tensor.semantic == Semantic.CATEGORICAL_SET:
      value = semantic_tensor.tensor
      if isinstance(value, tf.SparseTensor):
        value = tf.RaggedTensor.from_sparse(value)

      if semantic_tensor.tensor.dtype in FlexibleCategoricalSetStringTypes:
        normalized_inputs[key] = SemanticTensor(
            semantic=semantic_tensor.semantic, tensor=tf.cast(value, tf.string))
      elif semantic_tensor.tensor.dtype in FlexibleCategoricalSetIntTypes:
        normalized_inputs[key] = SemanticTensor(
            semantic=semantic_tensor.semantic,
            tensor=tf.cast(value, tf.int32) + CATEGORICAL_INTEGER_OFFSET)
      else:
        raise ValueError(
            "Non supported tensor dtype {} for semantic {} of feature {}"
            .format(semantic_tensor.tensor.dtype, semantic_tensor.semantic,
                    key))

    elif semantic_tensor.semantic == Semantic.HASH:
      if semantic_tensor.tensor.dtype in FlexibleHashTypes:
        _unroll_and_normalize(
            tf.cast(semantic_tensor.tensor, tf.string),
            semantic_tensor.semantic, key, "", normalized_inputs)
      else:
        raise ValueError(
            "Non supported tensor dtype {} for semantic {} of feature {}"
            .format(semantic_tensor.tensor.dtype, semantic_tensor.semantic,
                    key))

    elif semantic_tensor.semantic == Semantic.BOOLEAN:
      if semantic_tensor.tensor.dtype in FlexibleBooleanTypes:
        _unroll_and_normalize(
            tf.cast(semantic_tensor.tensor, tf.float32),
            semantic_tensor.semantic, key, math.nan, normalized_inputs)
      else:
        raise ValueError(
            "Non supported tensor dtype {} for semantic {} of feature {}"
            .format(semantic_tensor.tensor.dtype, semantic_tensor.semantic,
                    key))

    else:
      raise ValueError("Non supported semantic {} of feature {}".format(
          semantic_tensor.semantic, key))

  return normalized_inputs


def _density_tensor(value: AnyTensor, semantic: Semantic, base_key: str,
                    missing_value: Union[float, int, str]) -> tf.Tensor:
  """Density a possibly sparse tensor."""

  if isinstance(value, tf.SparseTensor) or isinstance(
      value, tf.compat.v1.SparseTensorValue):

    common_error = (
        "If the feature is multi-dimensional, use a static shape "
        "(or a tensor != sparse tensor). If the feature is a set, specify the "
        "set semantic manually.")

    if len(value.shape) != 2:
      raise ValueError("Expect rank 2 for tensor {}".format(value))

    if value.shape[1] is None:
      # The shape is not known at compilation time.
      tf.debugging.Assert(value.dense_shape[1] == 1, [
          "{} with tensor {} is provided as a "
          "sparse tensor with dynamic shape. Such feature can only be scalar "
          "but multiple values have been observed at the same time.".format(
              base_key, value)
      ])

    else:
      if value.shape[1] != 1:
        raise ValueError(
            "Invalid static shape for feature {} represented as sparse tensor "
            "{}. Expect 1, got {}. {}".format(base_key, value, value.shape[1],
                                              common_error))

    value = tf.sparse.to_dense(value, default_value=missing_value)

  elif isinstance(value, tf.RaggedTensor):
    raise ValueError(
        "Ragged tensors are not supported for feature {} with scalar type {} "
        "and semantic {}. If the feature is a set, specify the set semantic "
        "manually.".format(base_key, value, semantic))

  elif isinstance(value, tf.Tensor):
    pass  # Native format.

  else:
    raise ValueError("Unsupported tensor type: {}".format(value))

  return value


def _unroll_and_normalize(
    value: AnyTensor,
    semantic: Semantic,
    base_key: str,
    missing_value: Union[float, int, str],
    output: Dict[str, SemanticTensor],
    dense_preprocess: Optional[Callable[[tf.Tensor],
                                        tf.Tensor]] = None) -> None:
  """Splits a 2d tensor into a list of 1d tensors."""

  value = _density_tensor(value, semantic, base_key, missing_value)

  if dense_preprocess:
    value = dense_preprocess(value)

  rank = len(value.shape)

  if rank == 0:
    output[base_key] = SemanticTensor(
        semantic=semantic, tensor=tf.expand_dims(value, axis=0))
    return

  elif rank == 1:
    output[base_key] = SemanticTensor(semantic=semantic, tensor=value)
    return

  elif rank == 2:
    # Splits a 2d tensor into a list of 1d tensors.
    num_dims = _num_inner_dimension(value)
    if num_dims == 1:
      output[base_key] = SemanticTensor(semantic=semantic, tensor=value[:, 0])
    else:
      for dim_idx in range(num_dims):
        key = f"{base_key}.{dim_idx}"
        output[key] = SemanticTensor(
            semantic=semantic, tensor=value[:, dim_idx])
  else:
    raise Exception(
        "Invalid rank {} for feature {}. Example of value: {}".format(
            rank, base_key, value))


def normalize_inputs_regexp(name: Text) -> Text:
  """Gets a regular expression capturing the normalized columns of a features.

  Normalizing a features (see "normalize_inputs") can create multiple normalized
  features with names possibly different from the original feature. This
  function creates a regexp that will match those.

  Args:
    name: Name of a feature.

  Returns:
    Regular expression matching the normalized features generated by
    "normalize_inputs".
  """

  return "^" + re.escape(name) + r"(\.[0-9]+)?$"


def _num_inner_dimension(value: tf.Tensor) -> int:
  """Number of elements of the inner dimension of a dense 2d tensor.

  If rank is unknown at graph creation, returns 1 and returns a tensor with
  always at least one element on the inner dimension. Use "missing_value"
  as filling value.

  Args:
    value: Input dense tensor of rank 2.

  Returns:
    Number of elements in inner dimension.

  Raises:
    ValueError: Invalid input.
  """

  if len(value.shape) != 2:
    raise ValueError(f"{value} is not of rank 2")

  if value.shape[1] is None:
    # Unknown number of elements at graph creation time.
    return 1
  elif isinstance(value.shape[1], int):
    # TF v2 behavior due to tf.enable_v2_behavior.
    return value.shape[1]
  elif value.shape[1].value:
    # TF v1 behavior.
    return value.shape[1].value
  else:
    raise ValueError(f"Non supported tensor {value}")


def train(input_ids: List[str],
          label_id: str,
          weight_id: Optional[str],
          model_id: str,
          learner: str,
          task: Optional[TaskType] = Task.CLASSIFICATION,
          generic_hparms: Optional[
              abstract_learner_pb2.GenericHyperParameters] = None,
          ranking_group: Optional[str] = None,
          training_config: Optional[abstract_learner_pb2.TrainingConfig] = None,
          deployment_config: Optional[
              abstract_learner_pb2.DeploymentConfig] = None,
          guide: Optional[data_spec_pb2.DataSpecificationGuide] = None,
          model_dir: Optional[str] = None,
          keep_model_in_resource: Optional[bool] = True) -> tf.Operation:
  """Trains a model on the dataset accumulated by collect_training_examples.

  Args:
    input_ids: Ids/names of the input features.
    label_id: Id/name of the label feature.
    weight_id: Id/name of the weight feature.
    model_id: Id of the model.
    learner: Key of the learner.
    task: Task to solve.
    generic_hparms: Hyper-parameter of the learner.
    ranking_group: Id of the ranking group feature. Only for ranking.
    training_config: Training configuration.
    deployment_config: Deployment configuration (e.g. where to train the model).
    guide: Dataset specification guide.
    model_dir: If specified, export the trained model into this directory.
    keep_model_in_resource: If true, keep the model as a training model
      resource.

  Returns:
    The OP that trigger the training.
  """

  if generic_hparms is None:
    generic_hparms = abstract_learner_pb2.GenericHyperParameters()

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

  feature_ids = [_input_key_to_id(model_id, x) for x in sorted(input_ids)]

  if ranking_group is not None:
    training_config.ranking_group = ranking_group
    feature_ids.append(_input_key_to_id(model_id, ranking_group))

  return training_op.SimpleMLModelTrainer(
      feature_ids=",".join(feature_ids),
      label_id=_input_key_to_id(model_id, label_id),
      weight_id="" if weight_id is None else _input_key_to_id(
          model_id, weight_id),
      model_id=model_id if keep_model_in_resource else "",
      model_dir=model_dir or "",
      learner=learner,
      hparams=generic_hparms.SerializeToString(),
      task=task,
      training_config=training_config.SerializeToString(),
      deployment_config=deployment_config.SerializeToString(),
      guide=guide.SerializeToString())


def train_on_file_dataset(
    train_dataset_path: str,
    valid_dataset_path: Optional[str],
    feature_ids: List[str],
    label_id: str,
    weight_id: Optional[str],
    model_id: str,
    learner: str,
    task: Optional[TaskType] = Task.CLASSIFICATION,
    generic_hparms: Optional[
        abstract_learner_pb2.GenericHyperParameters] = None,
    ranking_group: Optional[str] = None,
    training_config: Optional[abstract_learner_pb2.TrainingConfig] = None,
    deployment_config: Optional[abstract_learner_pb2.DeploymentConfig] = None,
    guide: Optional[data_spec_pb2.DataSpecificationGuide] = None,
    model_dir: Optional[str] = None,
    keep_model_in_resource: Optional[bool] = True,
    working_cache_path: Optional[str] = None,
    distribution_config: Optional[DistributionConfiguration] = None
) -> tf.Operation:
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
    feature_ids: Ids/names of the input features.
    label_id: Id/name of the label feature.
    weight_id: Id/name of the weight feature.
    model_id: Id of the model.
    learner: Key of the learner.
    task: Task to solve.
    generic_hparms: Hyper-parameter of the learner.
    ranking_group: Id of the ranking group feature. Only for ranking.
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

  Returns:
    The OP that trigger the training.
  """

  if generic_hparms is None:
    generic_hparms = abstract_learner_pb2.GenericHyperParameters()

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

  if ranking_group is not None:
    training_config.ranking_group = ranking_group

  # Set the method argument into the proto configs.
  training_config.learner = learner
  training_config.task = task
  training_config.label = label_id

  if weight_id is not None:
    training_config.weight_definition.attribute = weight_id
    training_config.weight_definition.numerical.SetInParent()

  for feature_id in feature_ids:
    training_config.features.append(normalize_inputs_regexp(feature_id))

  if working_cache_path is not None:
    deployment_config.cache_path = working_cache_path

  if distribution_config is not None:
    deployment_config.try_resume_training = True
    deployment_config.distribute.implementation_key = "TF_DIST"

    if distribution_config.workers_addresses is not None:
      dst_addresses = deployment_config.distribute.Extensions[
          tf_distribution_pb2.tf_distribution].addresses
      dst_addresses.addresses[:] = distribution_config.workers_addresses

    else:
      # Assume the worker paths are provided through the env.
      deployment_config.distribute.Extensions[
          tf_distribution_pb2.tf_distribution].environment_variable.SetInParent(
          )

  return training_op.SimpleMLModelTrainerOnFile(
      train_dataset_path=train_dataset_path,
      valid_dataset_path=valid_dataset_path if valid_dataset_path else "",
      model_id=model_id if keep_model_in_resource else "",
      model_dir=model_dir or "",
      hparams=generic_hparms.SerializeToString(),
      training_config=training_config.SerializeToString(),
      deployment_config=deployment_config.SerializeToString(),
      guide=guide.SerializeToString())


def finalize_distributed_dataset_collection(cluster_coordinator,
                                            input_ids: List[str], model_id: str,
                                            dataset_path: str) -> None:
  """Finaliazes the collection of the partial dataset cache.

  Args:
    cluster_coordinator: Cluster coordinator of the distributed training.
    input_ids: Key of the input features and label to collect.
    model_id: Id of the model.
    dataset_path: Directory path to the output partial dataset cache.
  """

  sorted_input_ids = sorted(input_ids)
  feature_resource_ids = [
      _input_key_to_id(model_id, x) for x in sorted_input_ids
  ]

  def worker_fn():
    training_op.SimpleMLWorkerFinalizeFeatureOnFile(
        feature_resource_ids=feature_resource_ids, dataset_path=dataset_path)

  _execute_function_on_each_worker(cluster_coordinator, worker_fn)

  training_op.SimpleMLChiefFinalizeFeatureOnFile(
      feature_names=sorted_input_ids,
      num_shards=len(cluster_coordinator._cluster.workers),  # pylint: disable=protected-access
      dataset_path=dataset_path)


def _execute_function_on_each_worker(cluster_coordinator, call_fn):
  """Executes `call_fn` once on each of the workers."""

  # TODO(gbm): Call in parallel using multi-threading.
  # TODO(gbm): Test for stability when a worker is rescheduled.

  # pylint: disable=protected-access
  num_workers = cluster_coordinator._strategy._extended._num_workers
  for worker_idx in range(num_workers):
    with tf.device(f"/job:worker/task:{worker_idx}"):
      call_fn()
  # pylint: enable=protected-access


def _input_key_to_id(model_id: str, key: str) -> str:
  """Gets the name of the feature accumulator resource."""

  # Escape the commas that are used to separate the column resource id.
  # Those IDs have not impact to the final model, but they should be unique and
  # not contain commas.
  #
  # Turn the character '|' into an escape symbol.
  input_id = model_id + "_" + key.replace("|", "||").replace(",", "|c")
  if "," in input_id:
    raise ValueError(f"Internal error: Found comma in input_id {input_id}")
  return input_id


def combine_tensors_and_semantics(
    inputs: Dict[str, AnyTensor],
    semantics: Dict[str, Semantic]) -> Dict[str, SemanticTensor]:
  """Combines a mapping of tensors and a mapping of semantics.

  "inputs" should be a super-set (non strict) of "semantics". Items in "inputs"
  but not in "semantics" will be ignored.

  Args:
    inputs: Dictionary of tensors.
    semantics: Dictionary of semantics.

  Returns:
    A dictionary of semantic and guide tensors.
  """

  if not set(semantics.keys()).issubset(inputs.keys()):
    raise ValueError("semantics is not a subset of inputs "
                     "(inputs={} vs semantics={}).".format(
                         inputs.keys(), semantics.keys()))

  semantic_tensors = {}
  for key, semantic in semantics.items():
    semantic_tensors[key] = SemanticTensor(
        semantic=semantic, tensor=inputs[key])
  return semantic_tensors


def decombine_tensors_and_semantics(
    semantic_tensors: Dict[str, SemanticTensor]
) -> Tuple[Dict[str, AnyTensor], Dict[str, Semantic]]:
  """Splits the semantics and tensors of a mapping of semantic tensors.

  The inverse of "combine_tensors_and_semantics".

  Args:
    semantic_tensors: Dictionary of semantic tensors.

  Returns:
    A dictionary of tensors and a dictionary of semantics.
  """

  semantics = {}
  tensors = {}
  for key, value in semantic_tensors.items():
    tensors[key] = value.tensor
    semantics[key] = value.semantic
  return tensors, semantics


def infer_one_semantic(value: AnyTensor) -> Semantic:
  """Determines the most likely semantic of a tensor/Input.

  The core assumptions are:
    - Integers can both represent numerical and categorical values. Numerical is
      more common and less risky.
    - Sparse tensors are commonly used to represent possibly missing values and
      sets. The use for missing values is more common and less risky.
    - Ragged tensors are only used for represent sets.

  Args:
    value: Target tensor.

  Returns:
    Semantic of the target tensor.
  """

  dtype = value.dtype
  numerical_types = [
      float, np.float, np.int16, np.int32, np.int64, int, np.float32, np.float64
  ]

  if isinstance(value, tf.Tensor):
    if dtype in numerical_types:
      return Semantic.NUMERICAL
    else:
      return Semantic.CATEGORICAL
  elif isinstance(value, tf.SparseTensor):
    if dtype in numerical_types:
      return Semantic.NUMERICAL
    else:
      return Semantic.CATEGORICAL
  elif isinstance(value, tf.RaggedTensor):
    if dtype in [float, np.float, np.float32, np.float64]:
      raise ValueError("Only categorical-set features can be represented as a "
                       "ragged tensor. {} look numerical.".format(value))
    return Semantic.CATEGORICAL_SET

  raise ValueError(
      "Cannot infer semantic for tensor \"{}\" with dtype={}".format(
          value.name, dtype))


def infer_semantic(
    inputs: Dict[str, AnyTensor],
    manual_semantics: Optional[Dict[str, Optional[Semantic]]] = None,
    exclude_non_specified: Optional[bool] = False) -> Dict[str, Semantic]:
  """Determines the most likely semantic of a dict of tensors/Inputs."""

  manual_semantics = manual_semantics or {}
  for key in manual_semantics.keys():
    if key not in inputs:
      raise ValueError(
          f"Manual semantic \"{key}\" was not found among the input features: "
          f"{inputs.keys()}")

  semantics = {}
  for key, value in inputs.items():

    if exclude_non_specified and key not in manual_semantics:
      # Ignore this feature.
      continue

    # Manual semantic.
    semantic = manual_semantics.get(key, None)

    # Automated semantic.
    if semantic is None:
      try:
        semantic = infer_one_semantic(value)
      except ValueError as error:
        raise ValueError(
            f"Error when inferring the semantic of column \"{key}\"."
        ) from error

    semantics[key] = semantic

  return semantics


def infer_semantic_from_dataframe(dataset: pd.DataFrame) -> Dict[str, Semantic]:
  """Infers the semantic of the columns in a pandas dataframe."""

  semantics = {}
  for col in dataset.columns:
    dtype = dataset[col].dtype
    if dtype in [float, np.float, np.int16, np.int32, np.int64, int]:
      semantics[col] = Semantic.NUMERICAL
    elif dtype in [str, object]:
      semantics[col] = Semantic.CATEGORICAL
    else:
      raise Exception(
          "Cannot infer semantic for column \"{}\" with dtype={}".format(
              col, dtype))

  return semantics


def hparams_dict_to_generic_proto(
    hparams: Optional[HyperParameters] = None
) -> abstract_learner_pb2.GenericHyperParameters:
  """Converts hyper-parameters from dict to proto representation."""

  generic = abstract_learner_pb2.GenericHyperParameters()
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
          "Unsupported type \"{}:{}\" for hyper-parameter \"{}\". "
          "Possible types are int (for integer), float (for real), and str "
          "(for categorical)".format(value, type(value), key))

  return generic


def column_type_to_semantic(col_type: data_spec_pb2.ColumnType) -> Semantic:
  """Converts a column type into a semantic."""

  if col_type == data_spec_pb2.ColumnType.NUMERICAL:
    return Semantic.NUMERICAL

  if col_type == data_spec_pb2.ColumnType.CATEGORICAL:
    return Semantic.CATEGORICAL

  if col_type == data_spec_pb2.ColumnType.CATEGORICAL_SET:
    return Semantic.CATEGORICAL_SET

  if col_type == data_spec_pb2.ColumnType.BOOLEAN:
    return Semantic.BOOLEAN

  raise ValueError(f"Non conversion available for {col_type}")
