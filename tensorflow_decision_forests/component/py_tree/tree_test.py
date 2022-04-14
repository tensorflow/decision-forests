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

from absl import logging
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_decision_forests.component.py_tree import condition as condition_lib
from tensorflow_decision_forests.component.py_tree import dataspec as dataspec_lib
from tensorflow_decision_forests.component.py_tree import node as node_lib
from tensorflow_decision_forests.component.py_tree import tree as tree_lib
from tensorflow_decision_forests.component.py_tree import value as value_lib


class TreeTest(parameterized.TestCase, tf.test.TestCase):

  def test_empty_tree(self):
    tree = tree_lib.Tree(None)
    logging.info("Tree:\n%s", tree)

  def test_basic_tree(self):
    tree = tree_lib.Tree(
        node_lib.NonLeafNode(
            condition=condition_lib.NumericalHigherThanCondition(
                feature=dataspec_lib.SimpleColumnSpec(
                    name="f1", type=dataspec_lib.ColumnType.NUMERICAL),
                threshold=1.5,
                missing_evaluation=False),
            pos_child=node_lib.LeafNode(
                value=value_lib.RegressionValue(
                    value=5.0, num_examples=10, standard_deviation=1.0)),
            neg_child=node_lib.LeafNode(
                value=value_lib.ProbabilityValue(
                    probability=[0.5, 0.4, 0.1], num_examples=10))))

    tree_repr = repr(tree)
    logging.info("Tree repr:\n%s", tree_repr)
    # The "repr" is a single line that does not contain any line return.
    self.assertNotIn("\n", tree_repr)

    logging.info("Tree str:\n%s", tree)

    pretty = tree.pretty()
    logging.info("Pretty:\n%s", pretty)

    self.assertEqual(
        pretty, """(f1 >= 1.5; miss=False, score=None)
    ├─(pos)─ RegressionValue(value=5.0,sd=1.0,n=10)
    └─(neg)─ ProbabilityValue([0.5, 0.4, 0.1],n=10)
""")

  def test_stump_with_label_classes(self):
    tree = tree_lib.Tree(
        node_lib.LeafNode(
            value=value_lib.ProbabilityValue(
                probability=[0.5, 0.4, 0.1], num_examples=10)),
        label_classes=["a", "b", "c"])
    logging.info("Tree:\n%s", tree)


if __name__ == "__main__":
  tf.test.main()
