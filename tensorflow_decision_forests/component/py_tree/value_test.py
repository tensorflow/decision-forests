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

from tensorflow_decision_forests.component.py_tree import value as value_lib
from yggdrasil_decision_forests.model.decision_tree import decision_tree_pb2


class ValueTest(parameterized.TestCase, tf.test.TestCase):

  def test_regression(self):
    value = value_lib.RegressionValue(
        value=5.0, num_examples=10, standard_deviation=1.0
    )
    logging.info("value:\n%s", value)

  def test_probability(self):
    value = value_lib.ProbabilityValue(
        probability=[0.5, 0.4, 0.1], num_examples=10
    )
    logging.info("value:\n%s", value)

  def test_uplift(self):
    value = value_lib.UpliftValue(treatment_effect=[1, 2], num_examples=10)
    logging.info("value:\n%s", value)

  def test_core_value_to_value_classifier(self):
    core_node = decision_tree_pb2.Node()
    core_node.classifier.distribution.counts[:] = [0.0, 8.0, 2.0]
    core_node.classifier.distribution.sum = 10.0
    self.assertEqual(
        value_lib.core_value_to_value(core_node),
        value_lib.ProbabilityValue(probability=[0.8, 0.2], num_examples=10),
    )

  def test_core_value_to_value_regressor(self):
    core_node = decision_tree_pb2.Node()
    core_node.regressor.top_value = 1.0
    core_node.regressor.distribution.sum = 10.0
    core_node.regressor.distribution.sum_squares = 20.0
    core_node.regressor.distribution.count = 10.0
    self.assertEqual(
        value_lib.core_value_to_value(core_node),
        value_lib.RegressionValue(
            value=1.0, num_examples=10, standard_deviation=1.0
        ),
    )

  def test_core_value_to_value_uplift(self):
    core_node = decision_tree_pb2.Node()
    core_node.uplift.treatment_effect[:] = [0.0, 8.0, 2.0]
    core_node.uplift.sum_weights = 10.0
    self.assertEqual(
        value_lib.core_value_to_value(core_node),
        value_lib.UpliftValue(
            treatment_effect=[0.0, 8.0, 2.0], num_examples=10.0
        ),
    )


if __name__ == "__main__":
  tf.test.main()
