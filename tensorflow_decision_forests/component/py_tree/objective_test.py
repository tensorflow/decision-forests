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

from tensorflow_decision_forests.component.py_tree import objective as objective_lib

# pylint: disable=g-long-lambda


class ObjectiveTest(parameterized.TestCase, tf.test.TestCase):

  def test_classification(self):
    objective = objective_lib.ClassificationObjective(
        label="label", num_classes=5)
    logging.info("objective: %s", objective)

    objective = objective_lib.ClassificationObjective(
        label="label", classes=["a", "b"])
    logging.info("objective: %s", objective)

    objective = objective_lib.ClassificationObjective(
        label="label", classes=["a", "b"])
    logging.info("objective: %s", objective)

    objective = objective_lib.ClassificationObjective(
        label="label", classes=["a", "b"], num_classes=2)
    logging.info("objective: %s", objective)

  def test_classification_errors(self):
    self.assertRaises(
        ValueError,
        lambda: objective_lib.ClassificationObjective(label="label"))
    self.assertRaises(
        ValueError,
        lambda: objective_lib.ClassificationObjective(label="", num_classes=5))
    self.assertRaises(
        ValueError, lambda: objective_lib.ClassificationObjective(
            label="label", classes=["a", "b"], num_classes=5))
    self.assertRaises(
        ValueError, lambda: objective_lib.ClassificationObjective(
            label="label", classes=[]))

  def test_regression(self):
    objective = objective_lib.RegressionObjective(label="label")
    logging.info("objective: %s", objective)

  def test_ranking(self):
    objective = objective_lib.RankingObjective(label="label", group="group")
    logging.info("objective: %s", objective)


if __name__ == "__main__":
  tf.test.main()
