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

import os
from absl import flags
from absl import logging
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_decision_forests.component.model_plotter import model_plotter
from tensorflow_decision_forests.component.py_tree import condition as condition_lib
from tensorflow_decision_forests.component.py_tree import dataspec as dataspec_lib
from tensorflow_decision_forests.component.py_tree import node as node_lib
from tensorflow_decision_forests.component.py_tree import tree as tree_lib
from tensorflow_decision_forests.component.py_tree import value as value_lib


class ModelPlotterTest(parameterized.TestCase, tf.test.TestCase):

  def _save_plot(self, plot):
    plot_path = os.path.join(self.get_temp_dir(), "plot.html")
    logging.info("Plot to %s", plot_path)
    with open(plot_path, "w") as f:
      f.write(plot)

  def test_empty_tree(self):
    tree = tree_lib.Tree(None)
    plot = model_plotter.plot_tree(tree=tree)
    self._save_plot(plot)

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
    plot = model_plotter.plot_tree(
        tree=tree,
        display_options=model_plotter.DisplayOptions(node_x_size=150))
    self._save_plot(plot)

  def test_basic_tree_with_label_classes(self):
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
                    probability=[0.5, 0.4, 0.1], num_examples=10))),
        label_classes=["x", "y", "z"])
    plot = model_plotter.plot_tree(tree=tree)
    self._save_plot(plot)


if __name__ == "__main__":
  tf.test.main()
