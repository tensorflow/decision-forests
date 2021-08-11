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
from tensorflow_decision_forests.component.py_tree import value as value_lib

from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.model.decision_tree import decision_tree_pb2


class NodeTest(parameterized.TestCase, tf.test.TestCase):

  def test_leaf(self):
    node = node_lib.LeafNode(
        value=value_lib.RegressionValue(
            value=5.0, num_examples=10, standard_deviation=1.0))
    core_node = decision_tree_pb2.Node(
        regressor=decision_tree_pb2.NodeRegressorOutput(top_value=5.0))
    dist = core_node.regressor.distribution
    dist.count = 10.0
    dist.sum = 0
    dist.sum_squares = 10.0
    self.assertEqual(
        node_lib.node_to_core_node(node, data_spec_pb2.DataSpecification()),
        core_node)
    logging.info("node:\n%s", node)

  def test_non_leaf_without_children(self):
    node = node_lib.NonLeafNode(
        condition=condition_lib.NumericalHigherThanCondition(
            feature=dataspec_lib.SimpleColumnSpec(
                name="f1", type=dataspec_lib.ColumnType.NUMERICAL),
            threshold=1.5,
            missing_evaluation=False))
    logging.info("node:\n%s", node)

  def test_non_leaf_with_children(self):
    node = node_lib.NonLeafNode(
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
                probability=[0.5, 0.4, 0.1], num_examples=10)))
    logging.info("node:\n%s", node)


if __name__ == "__main__":
  tf.test.main()
