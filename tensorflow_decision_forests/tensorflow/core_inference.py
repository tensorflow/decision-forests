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

"""Core classes and functions of TensorFlow Decision Forests inference."""

import enum
import math
import re
from typing import Optional, Tuple, Dict, Union, NamedTuple, Callable, Any

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow_decision_forests.component.inspector import inspector as inspector_lib
from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.model import abstract_model_pb2

# The available tensor dtypes for each semantic.
FlexibleNumericalTypes = [
    tf.float32, tf.int8, tf.int16, tf.int32, tf.int64, tf.float64
]
FlexibleCategoricalIntTypes = [
    tf.int8,
    tf.int16,
    tf.int32,
    tf.int64,
    tf.float32,
]
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


class NodeFormat(enum.Enum):
  """Node format of  a model.

  Determines the storage format for nodes used by Yggdrasil Decision Forests.

  Attributes:
    BLOB_SEQUENCE: Sequence of blobs (list of bytes) stored in a byte stream.
  """
  BLOB_SEQUENCE = "BLOB_SEQUENCE"


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
      have the same meaning). Integer categorical values: (1) The training logic
      and model representation is optimized with the assumption that values are
      dense. (2) Internally, the value is stored as int32. The values should be
      <~2B. (3) The number of possible value is computed automatically from the
      training dataset. During inference, integer values greater than any value
      seen during training will be treated as out-of-vocabulary. (4) Minimum
      frequency and maximum vocabulary size constrains don't apply.
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
    DISCRETIZED_NUMERICAL: Numerical values automatically discretized into bins.
      Discretized numerical features are faster to train than (non-discretized)
      numerical features. If the number of unique values of these features is
      lower than the number of bins, the discretization is lossless from the
      point of view of the model. If the number of unique values of this
      features is greater than the number of bins, the discretization is lossy
      from the point of view of the model. Lossy discretization can reduce and
      sometime increase (due to regularization) the quality of the model.
  """

  NUMERICAL = 1
  CATEGORICAL = 2
  HASH = 3
  CATEGORICAL_SET = 4
  BOOLEAN = 5
  DISCRETIZED_NUMERICAL = 6


# Any tensorflow tensor.
AnyTensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]


class SemanticTensor(NamedTuple):
  """Combination of a tensor and its semantic."""
  semantic: Semantic
  tensor: AnyTensor = None


# Magic offset to add to categorical feature stored as integer.
# Yggdrasil reserves the integer value "0" of CATEGORICAL features for the
# "out of vocabulary" item. Since it is common to have integer categorical
# feature values using 0 as a normal value, Yggdrasil sees always the original
# value +1.
# Note: Negative integer value for categorical features are forbidden.
CATEGORICAL_INTEGER_OFFSET = 1

# The task of a model e.g. classification, regression, ranking.
Task = abstract_model_pb2.Task
TaskType = "abstract_model_pb2.Task"  # pylint: disable=invalid-name

# "InputModelSignatureFn" is a lambda that returns the
# (Dense,Sparse,Ragged)TensorSpec (or structure of TensorSpec e.g. dictionary,
# list) corresponding to input signature of the model.
#
# This is used when creating a model manually or when training with the
# "fit_on_file" method i.e. when no examples represent as tensors are available.
#
# If not specified, input model signatures are created with
# "build_default_input_model_signature".
#
# Overriding the input model signature is useful, for example, if a numerical
# input feature (which is consumed as DenseTensorSpec(float32) by default) is
# expected to be feed differently (e.g. RaggedTensor(int64)).
#
# The signature is as follow:
#
# def custom_model_input_signature(inspector):
#   input_spec = tfdf.keras.build_default_input_model_signature(inspector)
#   # "f1" is provided as an int64 (instead of a float32).
#   input_spec["f1"] = tf.TensorSpec(shape=[None], dtype=tf.int64)
#   return input_spec
#
InputModelSignatureFn = Callable[[inspector_lib.AbstractInspector], Any]


def normalize_inputs(
    inputs: Dict[str, SemanticTensor],
    categorical_integer_offset_correction: bool) -> Dict[str, SemanticTensor]:
  """Normalize input tensors for OP consumption.

  Normalization involves:
    - Casting the feature to the "normalized" dtype.
    - Dividing fixed-length features into sets of single dimensional features.
    - Converts the sparse tensors (generally used to represent possible missing
      values) into dense tensors with the correct missing value representation.

  Args:
    inputs: Dict of semantic tensor.
    categorical_integer_offset_correction: Apply a +1 offset on categorical
      integer features.

  Returns:
    Dict of normalized semantic tensors.

  Raises:
    ValueError: The arguments are invalid.
  """

  normalized_inputs = {}

  offset = (
      CATEGORICAL_INTEGER_OFFSET
      if categorical_integer_offset_correction else 0)

  for key, semantic_tensor in inputs.items():
    if semantic_tensor.semantic in [
        Semantic.NUMERICAL, Semantic.DISCRETIZED_NUMERICAL
    ]:
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
            -1 - offset,
            normalized_inputs,
            dense_preprocess=lambda x: x + offset)
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
            tensor=tf.cast(value, tf.int32) + offset)
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
      float, np.int8, np.int16, np.int32, np.int64, int, np.float32, np.float64
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
    if dtype in [float, np.float32, np.float64]:
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
    if dtype in [float, np.int16, np.int32, np.int64, int]:
      semantics[col] = Semantic.NUMERICAL
    elif dtype in [str, object]:
      semantics[col] = Semantic.CATEGORICAL
    else:
      raise Exception(
          "Cannot infer semantic for column \"{}\" with dtype={}".format(
              col, dtype))

  return semantics


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


def build_default_input_model_signature(
    inspector: inspector_lib.AbstractInspector) -> Any:

  tensor_specs = {}
  for feature in inspector.features():
    tensor_specs[feature.name] = build_default_feature_signature(
        feature.name, inspector.dataspec.columns[feature.col_idx])

  return tensor_specs


def build_default_feature_signature(
    feature_name: str, dataspec_column: data_spec_pb2.Column) -> Any:
  """Gets an example of feature values for the default model signature.

  When a model is trained without having tensor examples (e.g. the model is
  build manually or converted from the Yggdrasil format), the output of this
  function is used to instantiate the tf graph of the model. The feature value
  should represent 2 or more examples.

  Follow "InputModelSignatureFn".

  Args:
    feature_name: Name of the feature.
    dataspec_column: Yggdrasil column dataspec for the feature.

  Returns:
    An example of tensor value.
  """

  if dataspec_column.type == data_spec_pb2.ColumnType.NUMERICAL:
    return tf.TensorSpec(shape=[None], dtype=tf.float32)

  elif dataspec_column.type == data_spec_pb2.ColumnType.CATEGORICAL:
    if dataspec_column.categorical.is_already_integerized:
      return tf.TensorSpec(shape=[None], dtype=tf.int32)
    else:
      return tf.TensorSpec(shape=[None], dtype=tf.string)

  elif dataspec_column.type == data_spec_pb2.ColumnType.CATEGORICAL_SET:
    if dataspec_column.categorical.is_already_integerized:
      return tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32)
    else:
      return tf.RaggedTensorSpec(shape=[None, None], dtype=tf.string)

  elif dataspec_column.type == data_spec_pb2.ColumnType.BOOLEAN:
    return tf.TensorSpec(shape=[None], dtype=tf.float32)

  else:
    raise ValueError(
        f"Non supported feature type {feature_name}:{dataspec_column}")


def normalize_inputs_regexp(name: str, allow_index: bool) -> str:
  """Gets a regular expression capturing the normalized columns of a features.

  Normalizing a features (see "normalize_inputs") can create multiple normalized
  features with names possibly different from the original feature. This
  function creates a regexp that will match those.

  Args:
    name: Name of a feature.
    allow_index: If true, create a regex that captures unrolled
      multi-dimensional features.

  Returns:
    Regular expression matching the normalized features generated by
    "normalize_inputs".
  """

  if allow_index:
    return "^" + re.escape(name) + r"(\.[0-9]+)?$"
  else:
    return "^" + re.escape(name) + "$"
