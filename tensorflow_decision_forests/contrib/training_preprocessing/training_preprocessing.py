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

"""Utility for jointly preprocessing training data, labels and sample weights."""

import math
from typing import Callable, Dict, List, Tuple, Union

import tensorflow as tf
import tensorflow_decision_forests as tfdf


TensorDict = Dict[str, Union[tf.Tensor, tf.SparseTensor]]
TensorOrTensorDict = Union[tf.Tensor, TensorDict]

_CLASSIFICATION_MISSING_LABEL_VALUE = -2
_REGRESSION_MISSING_LABEL_VALUE = math.nan


def multitask_label_presence_processing(
    multitask_items: List[tfdf.keras.MultiTaskItem],
) -> Callable[
    [TensorOrTensorDict, TensorOrTensorDict, TensorOrTensorDict],
    Tuple[TensorOrTensorDict, TensorOrTensorDict, tf.Tensor],
]:
  """Returns a preprocessor for missing label imputation in case of Multitask models.

  TODO Add examples.

  Args:
    multitask_items: A list of multi-task configurations.

  Returns:
    A function that can be used for preprocessing labels and sample weights
    during a multi-task model training.
  """
  task_types = {}
  for multtask_item in multitask_items:
    task_types[multtask_item.label] = multtask_item.task

  # pytype: disable=bad-return-type
  @tf.function
  def label_presence_preprocessing(
      train_x: TensorOrTensorDict,
      train_y: TensorOrTensorDict,
      train_weights: TensorOrTensorDict,
  ) -> Tuple[TensorOrTensorDict, TensorOrTensorDict, tf.Tensor]:
    """A preprocessing fn for preprocessing labels and sample weights.

    Args:
      train_x: The training input features.
      train_y: The training labels.
      train_weights: The training sample weights.

    Returns:
      train_x: Unchanged training input features.
      processed_train_y: The labels with  imputed missing value.
      processed_train_weight: The sample weight where the per-task sample
        weights are reduced into a single tensor.

    Raises:
      ValueError: if both train_y or train_weights are not dicts.
      KeyError: if there is a mismatch between the task names specified in the
        multitask_items vs the task names in the label and weights.
    """
    if not isinstance(train_y, dict) or not isinstance(train_weights, dict):
      raise ValueError(
          'The preprocessor expects the label and sample_weights to be'
          ' dictionaries.'
      )
    # pytype: enable=bad-return-type

    processed_train_y = {}
    for task_name, task_type in task_types.items():
      if task_name not in train_y:
        raise KeyError(
            f'Task {task_name} was not found in train_y: {train_y.keys()}'
        )
      if task_name not in train_weights:
        raise KeyError(
            f'Task {task_name} was not found in train_weights:'
            f' {train_weights.keys()}'
        )

      processed_train_y[task_name] = _label_preprocessing(
          train_y[task_name], train_weights[task_name], task_type
      )
      processed_train_weight = _sample_weight_preprocessing(train_weights)

    return train_x, processed_train_y, processed_train_weight

  return label_presence_preprocessing


def _label_preprocessing(
    label: tf.Tensor, sample_weight: tf.Tensor, task: tfdf.keras.Task
) -> tf.Tensor:
  """Impute missing label values with TFDF special values.

  TFDF understands missing labels by looking at the values.
  For classification the special value that indicates a missing label is -2.
  For regression the special value that indicates a missing label is NaN.

  Args:
    label: the label tensor.
    sample_weight: the training sample weight tensor.
    task: the task type.

  Returns:
    The label tensor with missing values replaced by TFDF special values.

  Raises:
    NotImplementedError when a task is not CLASSIFICATION or REGRESSION.
  """
  if task == tfdf.keras.Task.CLASSIFICATION:
    # Replace missing labels with -2, which is the understood by TFDF as
    # missing.
    return tf.where(
        sample_weight == tf.cast(0.0, sample_weight.dtype),
        tf.cast(_CLASSIFICATION_MISSING_LABEL_VALUE, label.dtype),
        label,
    )
  elif task == tfdf.keras.Task.REGRESSION:
    # Replace missing labels with NaN, which is the understood by TFDF as
    # missing.
    return tf.where(
        sample_weight == tf.cast(0.0, sample_weight.dtype),
        tf.cast(_REGRESSION_MISSING_LABEL_VALUE, label.dtype),
        label,
    )
  else:
    raise NotImplementedError(
        'Lablel presence preprocessing only supports classification and'
        ' regression tasks.'
    )


def _sample_weight_preprocessing(
    sample_weights: Dict[str, tf.Tensor]
) -> tf.Tensor:
  """Reduce per-task sample weight into a single sample weight column."""
  sample_weights = tf.stack(tf.nest.flatten(sample_weights), axis=-1)
  return tf.reduce_max(sample_weights, axis=-1)
