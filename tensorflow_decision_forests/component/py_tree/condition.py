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

"""Conditions / splits for non-leaf nodes.

A condition (e.g. a>0.5) is evaluated to a binary value (e.g. True if a=5).
Condition evaluations control the branching of an example in a tree.
"""

import abc
from typing import List, Union, Optional

import six

from tensorflow_decision_forests.component.py_tree import dataspec as dataspec_lib
from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.model.decision_tree import decision_tree_pb2

ColumnType = data_spec_pb2.ColumnType
SimpleColumnSpec = dataspec_lib.SimpleColumnSpec


@six.add_metaclass(abc.ABCMeta)
class AbstractCondition(object):
  """Generic condition.

  Attrs:
    missing_evaluation: Result of the evaluation of the condition if the feature
      is missing. If None, a feature cannot be missing or a specific method run
      during inference to handle missing values.
  """

  def __init__(self,
               missing_evaluation: Optional[bool],
               split_score: Optional[float] = None):
    self._missing_evaluation = missing_evaluation
    self._split_score = split_score

  @property
  def missing_evaluation(self):
    return self._missing_evaluation

  @property
  def split_score(self) -> Optional[float]:
    return self._split_score

  @abc.abstractmethod
  def features(self) -> List[SimpleColumnSpec]:
    """List of features used to evaluate the condition."""

    pass

  def __repr__(self):
    return (f"AbstractCondition({self.features()}, "
            f"missing_evaluation={self._missing_evaluation}, "
            f"split_score={self._split_score})")


class IsMissingInCondition(AbstractCondition):
  """Condition of the form "attribute is missing"."""

  def __init__(self,
               feature: SimpleColumnSpec,
               split_score: Optional[float] = None):
    super(IsMissingInCondition, self).__init__(
        missing_evaluation=False, split_score=split_score
    )
    self._feature = feature

  def features(self):
    return [self._feature]

  def __repr__(self):
    return f"({self._feature.name} is missing, score={self._split_score})"

  def __eq__(self, other):
    if not isinstance(other, IsMissingInCondition):
      return False
    return self._feature == other._feature

  @property
  def feature(self):
    return self._feature


class IsTrueCondition(AbstractCondition):
  """Condition of the form "attribute is true"."""

  def __init__(self,
               feature: SimpleColumnSpec,
               missing_evaluation: Optional[bool],
               split_score: Optional[float] = None):
    super(IsTrueCondition, self).__init__(missing_evaluation, split_score)
    self._feature = feature

  def features(self):
    return [self._feature]

  def __repr__(self):
    return (f"({self._feature.name} is true; miss={self.missing_evaluation}, "
            f"score={self._split_score})")

  def __eq__(self, other):
    if not isinstance(other, IsTrueCondition):
      return False
    return self._feature == other._feature

  @property
  def feature(self):
    return self._feature


class NumericalHigherThanCondition(AbstractCondition):
  """Condition of the form "attribute >= threhsold"."""

  def __init__(self,
               feature: SimpleColumnSpec,
               threshold: float,
               missing_evaluation: Optional[bool],
               split_score: Optional[float] = None):
    super(NumericalHigherThanCondition, self).__init__(missing_evaluation,
                                                       split_score)
    self._feature = feature
    self._threshold = threshold

  def features(self):
    return [self._feature]

  def __repr__(self):
    return (f"({self._feature.name} >= {self._threshold}; "
            f"miss={self.missing_evaluation}, "
            f"score={self._split_score})")

  def __eq__(self, other):
    if not isinstance(other, NumericalHigherThanCondition):
      return False
    return (self._feature == other._feature and
            self._threshold == other._threshold)

  @property
  def feature(self):
    return self._feature

  @property
  def threshold(self):
    return self._threshold


class CategoricalIsInCondition(AbstractCondition):
  """Condition of the form "attribute in [...set of items...]"."""

  def __init__(self,
               feature: SimpleColumnSpec,
               mask: Union[List[str], List[int]],
               missing_evaluation: Optional[bool],
               split_score: Optional[float] = None):
    super(CategoricalIsInCondition, self).__init__(missing_evaluation,
                                                   split_score)
    self._feature = feature
    self._mask = mask

  def features(self):
    return [self._feature]

  def __repr__(self):
    return (f"({self._feature.name} in {self._mask}; "
            f"miss={self.missing_evaluation}, "
            f"score={self._split_score})")

  def __eq__(self, other):
    if not isinstance(other, CategoricalIsInCondition):
      return False
    return self._feature == other._feature and self._mask == other._mask

  @property
  def feature(self):
    return self._feature

  @property
  def mask(self):
    return self._mask


class CategoricalSetContainsCondition(AbstractCondition):
  """Condition of the form "attribute intersect [...set of items...]!=empty"."""

  def __init__(self,
               feature: SimpleColumnSpec,
               mask: Union[List[str], List[int]],
               missing_evaluation: Optional[bool],
               split_score: Optional[float] = None):
    super(CategoricalSetContainsCondition,
          self).__init__(missing_evaluation, split_score)
    self._feature = feature
    self._mask = mask

  def features(self):
    return [self._feature]

  def __repr__(self):
    return (f"({self._feature.name} intersect {self._mask} != empty; "
            f"miss={self.missing_evaluation}, "
            f"score={self._split_score})")

  def __eq__(self, other):
    if not isinstance(other, CategoricalSetContainsCondition):
      return False
    return self._feature == other._feature and self._mask == other._mask

  @property
  def feature(self):
    return self._feature

  @property
  def mask(self):
    return self._mask


class NumericalSparseObliqueCondition(AbstractCondition):
  """Condition of the form "attributes * weights >= threshold"."""

  def __init__(self,
               features: List[SimpleColumnSpec],
               weights: List[float],
               threshold: float,
               missing_evaluation: Optional[bool],
               split_score: Optional[float] = None):
    super(NumericalSparseObliqueCondition,
          self).__init__(missing_evaluation, split_score)
    self._features = features
    self._weights = weights
    self._threshold = threshold

  def features(self):
    return self._features

  def __repr__(self):
    return (f"({self._features} . {self._weights} >= {self._threshold}; "
            f"miss={self.missing_evaluation}, "
            f"score={self._split_score})")

  def __eq__(self, other):
    if not isinstance(other, NumericalSparseObliqueCondition):
      return False
    return (self._features == other._features and
            self._weights == other._weights and
            self._threshold == other._threshold)

  @property
  def weights(self):
    return self._weights

  @property
  def threshold(self):
    return self._threshold


def core_condition_to_condition(
    core_condition: decision_tree_pb2.NodeCondition,
    dataspec: data_spec_pb2.DataSpecification) -> AbstractCondition:
  """Converts a condition from the core to python format."""

  condition_type = core_condition.condition
  attribute = dataspec_lib.make_simple_column_spec(dataspec,
                                                   core_condition.attribute)
  column_spec = dataspec.columns[core_condition.attribute]

  if condition_type.HasField("na_condition"):
    return IsMissingInCondition(attribute, core_condition.split_score)

  if condition_type.HasField("higher_condition"):
    return NumericalHigherThanCondition(
        attribute, condition_type.higher_condition.threshold,
        core_condition.na_value, core_condition.split_score)

  if condition_type.HasField("true_value_condition"):
    return IsTrueCondition(attribute, core_condition.na_value,
                           core_condition.split_score)

  if condition_type.HasField("contains_bitmap_condition"):
    items = column_spec_bitmap_to_items(
        dataspec.columns[core_condition.attribute],
        condition_type.contains_bitmap_condition.elements_bitmap)
    if attribute.type == ColumnType.CATEGORICAL:
      return CategoricalIsInCondition(attribute, items, core_condition.na_value,
                                      core_condition.split_score)
    elif attribute.type == ColumnType.CATEGORICAL_SET:
      return CategoricalSetContainsCondition(attribute, items,
                                             core_condition.na_value,
                                             core_condition.split_score)

  if condition_type.HasField("contains_condition"):
    items = condition_type.contains_condition.elements
    if not column_spec.categorical.is_already_integerized:
      items = [
          dataspec_lib.categorical_value_idx_to_value(column_spec, item)
          for item in items
      ]
    if attribute.type == ColumnType.CATEGORICAL:
      return CategoricalIsInCondition(attribute, items, core_condition.na_value,
                                      core_condition.split_score)
    elif attribute.type == ColumnType.CATEGORICAL_SET:
      return CategoricalSetContainsCondition(attribute, items,
                                             core_condition.na_value,
                                             core_condition.split_score)

  if condition_type.HasField("discretized_higher_condition"):
    threshold = dataspec_lib.discretized_numerical_to_numerical(
        column_spec, condition_type.discretized_higher_condition.threshold)
    return NumericalHigherThanCondition(attribute, threshold,
                                        core_condition.na_value,
                                        core_condition.split_score)

  if condition_type.HasField("oblique_condition"):
    attributes = [
        dataspec_lib.make_simple_column_spec(dataspec, attribute_idx)
        for attribute_idx in condition_type.oblique_condition.attributes
    ]
    return NumericalSparseObliqueCondition(
        attributes, list(condition_type.oblique_condition.weights),
        condition_type.oblique_condition.threshold, core_condition.na_value,
        core_condition.split_score)

  raise ValueError(f"Non supported condition type: {core_condition}")


def column_spec_bitmap_to_items(column_spec: data_spec_pb2.Column,
                                bitmap: bytes) -> Union[List[int], List[str]]:
  """Converts a mask-bitmap into a list of elements."""

  items = []
  for value_idx in range(column_spec.categorical.number_of_unique_values):
    byte_idx = value_idx // 8
    sub_bit_idx = value_idx & 7
    has_item = (bitmap[byte_idx] & (1 << sub_bit_idx)) != 0
    if has_item:
      items.append(
          dataspec_lib.categorical_value_idx_to_value(column_spec, value_idx))

  return items


def column_spec_items_to_bitmap(column_spec: data_spec_pb2.Column,
                                items: List[int]) -> bytes:
  """Converts a list of elements into a mask-bitmap."""

  # Allocate a zero-bitmap.
  bitmap = bytearray(
      b"\0" * ((column_spec.categorical.number_of_unique_values + 7) // 8))
  for item in items:
    bitmap[item // 8] |= 1 << (item & 7)
  return bytes(bitmap)


def set_core_node(condition: AbstractCondition,
                  dataspec: data_spec_pb2.DataSpecification,
                  core_node: decision_tree_pb2.Node):
  """Sets a core node (proto format) from a python value."""

  core_condition = core_node.condition
  core_condition.na_value = condition.missing_evaluation

  if condition.split_score is not None:
    core_condition.split_score = condition.split_score

  features = condition.features()
  if not features:
    raise ValueError("Condition without features")
  core_condition.attribute = dataspec_lib.column_name_to_column_idx(
      features[0].name, dataspec)
  feature_column = dataspec.columns[core_condition.attribute]

  if isinstance(condition, IsMissingInCondition):
    core_condition.condition.na_condition.SetInParent()

  elif isinstance(condition, IsTrueCondition):
    core_condition.condition.true_value_condition.SetInParent()

  elif isinstance(condition, NumericalHigherThanCondition):
    core_condition.condition.higher_condition.threshold = condition.threshold

  elif isinstance(condition,
                  (CategoricalIsInCondition, CategoricalSetContainsCondition)):
    mask = condition.mask
    if mask and isinstance(mask[0], str):
      # Converts the mask to a list of integers.
      mask = [feature_column.categorical.items[value].index for value in mask]

    # Select the most efficient way to represent the mask
    if len(mask) * 32 * 8 > feature_column.categorical.number_of_unique_values:
      # A bitmap is more efficient.
      core_condition.condition.contains_bitmap_condition.elements_bitmap = column_spec_items_to_bitmap(
          feature_column, mask)
    else:
      # A list of indices is more efficient.
      core_condition.condition.contains_condition.elements[:] = mask

  elif isinstance(condition, NumericalSparseObliqueCondition):
    oblique = core_condition.condition.oblique_condition
    oblique.attributes[:] = [
        dataspec_lib.column_name_to_column_idx(feature.name, dataspec)
        for feature in features
    ]
    oblique.weights[:] = condition.weights
    oblique.threshold = condition.threshold

  else:
    raise NotImplementedError("No supported value type")
