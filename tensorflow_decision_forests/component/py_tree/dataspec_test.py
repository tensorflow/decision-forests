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

import math
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_decision_forests.component.py_tree import dataspec as dataspec_lib
from yggdrasil_decision_forests.dataset import data_spec_pb2


def toy_dataspec():
  dataspec = data_spec_pb2.DataSpecification()

  f1 = dataspec.columns.add()
  f1.name = "f1"
  f1.type = data_spec_pb2.ColumnType.NUMERICAL

  f2 = dataspec.columns.add()
  f2.name = "f2"
  f2.type = data_spec_pb2.ColumnType.CATEGORICAL
  f2.categorical.number_of_unique_values = 3
  f2.categorical.items["<OOD>"].index = 0
  f2.categorical.items["x"].index = 1
  f2.categorical.items["y"].index = 2

  f3 = dataspec.columns.add()
  f3.name = "f3"
  f3.type = data_spec_pb2.ColumnType.CATEGORICAL
  f3.categorical.number_of_unique_values = 3
  f3.categorical.is_already_integerized = True

  f4 = dataspec.columns.add()
  f4.name = "f4"
  f4.type = data_spec_pb2.ColumnType.DISCRETIZED_NUMERICAL
  f4.discretized_numerical.boundaries[:] = [0, 1, 2]
  return dataspec


class DataspecTest(parameterized.TestCase, tf.test.TestCase):

  def test_make_simple_column_spec(self):
    self.assertEqual(
        dataspec_lib.make_simple_column_spec(toy_dataspec(), 0),
        dataspec_lib.SimpleColumnSpec(
            name="f1", type=data_spec_pb2.ColumnType.NUMERICAL, col_idx=0))

  def test_categorical_value_idx_to_value(self):
    dataspec = toy_dataspec()
    self.assertEqual(
        dataspec_lib.categorical_value_idx_to_value(dataspec.columns[1], 1),
        "x")

    self.assertEqual(
        dataspec_lib.categorical_value_idx_to_value(dataspec.columns[2], 1), 1)

  def test_discretized_numerical_to_numerical(self):
    column_spec = toy_dataspec().columns[3]
    self.assertEqual(
        dataspec_lib.discretized_numerical_to_numerical(column_spec, 0), 0 - 1)
    self.assertEqual(
        dataspec_lib.discretized_numerical_to_numerical(column_spec, 1), 0.5)
    self.assertEqual(
        dataspec_lib.discretized_numerical_to_numerical(column_spec, 2), 1.5)
    self.assertEqual(
        dataspec_lib.discretized_numerical_to_numerical(column_spec, 3), 2 + 1)
    self.assertTrue(
        math.isnan(
            dataspec_lib.discretized_numerical_to_numerical(column_spec, 4)))

  def test_categorical_column_dictionary_to_list(self):
    dataspec = toy_dataspec()
    self.assertEqual(
        dataspec_lib.categorical_column_dictionary_to_list(dataspec.columns[1]),
        ["<OOD>", "x", "y"])

  def test_column_name_to_column_idx(self):
    dataspec = toy_dataspec()
    self.assertEqual(dataspec_lib.column_name_to_column_idx("f1", dataspec), 0)
    self.assertEqual(dataspec_lib.column_name_to_column_idx("f2", dataspec), 1)


if __name__ == "__main__":
  tf.test.main()
