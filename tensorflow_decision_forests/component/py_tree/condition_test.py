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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_decision_forests.component.py_tree import condition as condition_lib
from tensorflow_decision_forests.component.py_tree import dataspec as dataspec_lib
from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.model.decision_tree import decision_tree_pb2


class ConditionTest(parameterized.TestCase, tf.test.TestCase):

  def test_column_spec_bitmap_to_items_integer(self):
    column_spec = data_spec_pb2.Column()
    column_spec.categorical.number_of_unique_values = 10
    column_spec.categorical.is_already_integerized = True
    # b1100101101 => 32Dh
    self.assertEqual(
        condition_lib.column_spec_bitmap_to_items(column_spec, b"\x2D\x03"),
        [0, 2, 3, 5, 8, 9])

  def test_column_spec_bitmap_to_items_string(self):
    column_spec = data_spec_pb2.Column()
    column_spec.categorical.number_of_unique_values = 10
    for i in range(10):
      column_spec.categorical.items[f"item_{i}"].index = i
    column_spec.categorical.is_already_integerized = False
    # 1100101101b => 32Dh
    self.assertEqual(
        condition_lib.column_spec_bitmap_to_items(column_spec, b"\x2D\x03"),
        ["item_0", "item_2", "item_3", "item_5", "item_8", "item_9"])

  def test_core_condition_to_condition_is_missing(self):
    core_condition = decision_tree_pb2.NodeCondition()
    core_condition.na_value = False
    core_condition.attribute = 0
    core_condition.condition.na_condition.SetInParent()

    dataspec = data_spec_pb2.DataSpecification()
    column_spec = dataspec.columns.add()
    column_spec.name = "a"
    column_spec.type = dataspec_lib.ColumnType.NUMERICAL

    attribute = dataspec_lib.SimpleColumnSpec("a",
                                              dataspec_lib.ColumnType.NUMERICAL,
                                              0)
    self.assertEqual(
        condition_lib.core_condition_to_condition(core_condition, dataspec),
        condition_lib.IsMissingInCondition(attribute))


if __name__ == "__main__":
  tf.test.main()
