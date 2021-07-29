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

"""Utility for the dataset specification.

A dataset specification defines how to interpret the attributes/features in a
dataset.
"""

import math
from typing import NamedTuple, Union, Optional, List

from yggdrasil_decision_forests.dataset import data_spec_pb2

ColumnType = data_spec_pb2.ColumnType

# Special value to out of vocabulary items.
OUT_OF_DICTIONARY = "<OOD>"


class SimpleColumnSpec(NamedTuple):
  """Simplified representation of a column spec.

  For the (full) column spec, use "data_spec_pb2.columns[i]" directly.
  """

  # Name of the column.
  name: str
  # Semantic of the column.
  type: "ColumnType"
  # Index of the column in "data_spec_pb2.columns".
  col_idx: Optional[int] = None

  def __repr__(self):
    return f"\"{self.name}\" ({self.type}; #{self.col_idx})"


def make_simple_column_spec(dataspec: data_spec_pb2.DataSpecification,
                            col_idx: int) -> SimpleColumnSpec:
  """Creates a SimpleColumnSpec from a (full) DataSpecification."""

  column_spec = dataspec.columns[col_idx]
  return SimpleColumnSpec(column_spec.name, column_spec.type, col_idx)


def categorical_value_idx_to_value(column_spec: data_spec_pb2.Column,
                                   value_idx: int) -> Union[int, str]:
  """Gets the representation value of a categorical value stored as integer.

  If the categorical value is an integer, returns the input value.
  If the categorical value is a string, resolves the dictionary and returns a
  string.

  Args:
    column_spec: The column spec of the attribute.
    value_idx: A value compatible with the column spec.

  Returns:
    The representation of "value_idx".
  """

  if column_spec.categorical.is_already_integerized:
    return value_idx
  else:
    for key, value in column_spec.categorical.items.items():
      if value.index == value_idx:
        return key
    return OUT_OF_DICTIONARY


def categorical_column_dictionary_to_list(
    column_spec: data_spec_pb2.Column) -> List[str]:
  """Extracts the dictionary elements of a categorical column.

  Fails if the column does not contains a dictionary, or if the dictionary is
  incomplete.

  Args:
    column_spec: Dataspec column.

  Returns:
    The list of items.
  """

  if column_spec.categorical.is_already_integerized:
    raise ValueError("The column is pre-integerized and does not contain "
                     "a dictionary.")

  items = [None] * column_spec.categorical.number_of_unique_values

  for key, value in column_spec.categorical.items.items():
    if items[value.index] is not None:
      raise ValueError(f"Duplicated index {value.index} in dictionary")
    items[value.index] = key

  for index, value in enumerate(items):
    if value is None:
      raise ValueError(f"Invalid dictionary. Non value for index {index} "
                       f"in column {column_spec}")

  return items  # pytype: disable=bad-return-type


def label_value_idx_to_value(column_spec: data_spec_pb2.Column,
                             value_idx: int) -> Union[int, str]:
  """Gets the representation value of a categorical label value.

  This function can be used in condition with ProbabilityValue.

  Args:
    column_spec: The column spec of the attribute.
    value_idx: A label value compatible with the column spec.

  Returns:
    The representation of "value_idx".
  """

  # In the dataspec, the value "0" is reserved for the "Out-of-vocabulary" item.
  # Target label don't have such item.
  return categorical_value_idx_to_value(column_spec, value_idx + 1)


def discretized_numerical_to_numerical(column_spec: data_spec_pb2.Column,
                                       value: int) -> float:
  """Converts a discretized numerical value to a matching numerical value."""

  boundaries = column_spec.discretized_numerical.boundaries
  if value > len(boundaries):
    # Missing value
    return math.nan

  if value == 0:
    return boundaries[0] - 1.0

  if value == len(boundaries):
    # Missing value
    return boundaries[-1] + 1.0

  return (boundaries[value] + boundaries[value - 1]) / 2


def column_name_to_column_idx(name: str,
                              dataspec: data_spec_pb2.DataSpecification) -> int:
  """Gets a column index from its name."""

  for idx, column in enumerate(dataspec.columns):
    if name == column.name:
      return idx
  raise ValueError(f"Unknown column {name}")
