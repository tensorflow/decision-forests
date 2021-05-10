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
import os

from absl import flags
from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_decision_forests.component import py_tree
from tensorflow_decision_forests.component.builder import builder as builder_lib

Tree = py_tree.tree.Tree
NonLeafNode = py_tree.node.NonLeafNode
NumericalHigherThanCondition = py_tree.condition.NumericalHigherThanCondition
CategoricalIsInCondition = py_tree.condition.CategoricalIsInCondition
SimpleColumnSpec = py_tree.dataspec.SimpleColumnSpec
LeafNode = py_tree.node.LeafNode
ProbabilityValue = py_tree.value.ProbabilityValue
RegressionValue = py_tree.value.RegressionValue

# pylint: disable=g-long-lambda


def data_root_path() -> str:
  return ""


def test_data_path() -> str:
  return os.path.join(data_root_path(),
                      "external/ydf/yggdrasil_decision_forests/test_data")


def tmp_path() -> str:
  return flags.FLAGS.test_tmpdir


def test_model_directory() -> str:
  return os.path.join(test_data_path(), "model")


def test_dataset_directory() -> str:
  return os.path.join(test_data_path(), "model")


class BuilderTest(parameterized.TestCase, tf.test.TestCase):

  def test_classification_random_forest(self):
    model_path = os.path.join(tmp_path(), "classification_rf")
    logging.info("Create model in %s", model_path)
    builder = builder_lib.RandomForestBuilder(
        path=model_path,
        model_format=builder_lib.ModelFormat.TENSORFLOW_SAVED_MODEL,
        objective=py_tree.objective.ClassificationObjective(
            label="color", classes=["red", "blue", "green"]))

    #  f1>=1.5
    #    │
    #    ├─(pos)─ f2 in ["cat","dog"]
    #    │         │
    #    │         ├─(pos)─ value: [0.8, 0.1, 0.1]
    #    │         └─(neg)─ value: [0.1, 0.8, 0.1]
    #    └─(neg)─ value: [0.1, 0.1, 0.8]
    builder.add_tree(
        Tree(
            NonLeafNode(
                condition=NumericalHigherThanCondition(
                    feature=SimpleColumnSpec(
                        name="f1", type=py_tree.dataspec.ColumnType.NUMERICAL),
                    threshold=1.5,
                    missing_evaluation=False),
                pos_child=NonLeafNode(
                    condition=CategoricalIsInCondition(
                        feature=SimpleColumnSpec(
                            name="f2",
                            type=py_tree.dataspec.ColumnType.CATEGORICAL),
                        mask=["cat", "dog"],
                        missing_evaluation=False),
                    pos_child=LeafNode(
                        value=ProbabilityValue(
                            probability=[0.8, 0.1, 0.1], num_examples=10)),
                    neg_child=LeafNode(
                        value=ProbabilityValue(
                            probability=[0.1, 0.8, 0.1], num_examples=20))),
                neg_child=LeafNode(
                    value=ProbabilityValue(
                        probability=[0.1, 0.1, 0.8], num_examples=30)))))

    builder.close()

    logging.info("Loading model")
    loaded_model = tf.keras.models.load_model(model_path)

    logging.info("Make predictions")
    tf_dataset = tf.data.Dataset.from_tensor_slices({
        "f1": [1.0, 2.0, 3.0],
        "f2": ["cat", "cat", "bird"]
    }).batch(2)
    predictions = loaded_model.predict(tf_dataset)
    self.assertAllClose(predictions,
                        [[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])

  def test_classification_cart(self):
    model_path = os.path.join(tmp_path(), "classification_cart")
    logging.info("Create model in %s", model_path)
    builder = builder_lib.CARTBuilder(
        path=model_path,
        model_format=builder_lib.ModelFormat.TENSORFLOW_SAVED_MODEL,
        objective=py_tree.objective.ClassificationObjective(
            label="color", classes=["red", "blue", "green"]))

    #  f1>=1.5
    #    ├─(pos)─ f2 in ["cat","dog"]
    #    │         ├─(pos)─ value: [0.8, 0.1, 0.1]
    #    │         └─(neg)─ value: [0.1, 0.8, 0.1]
    #    └─(neg)─ value: [0.1, 0.1, 0.8]
    builder.add_tree(
        Tree(
            NonLeafNode(
                condition=NumericalHigherThanCondition(
                    feature=SimpleColumnSpec(
                        name="f1", type=py_tree.dataspec.ColumnType.NUMERICAL),
                    threshold=1.5,
                    missing_evaluation=False),
                pos_child=NonLeafNode(
                    condition=CategoricalIsInCondition(
                        feature=SimpleColumnSpec(
                            name="f2",
                            type=py_tree.dataspec.ColumnType.CATEGORICAL),
                        mask=["cat", "dog"],
                        missing_evaluation=False),
                    pos_child=LeafNode(
                        value=ProbabilityValue(
                            probability=[0.8, 0.1, 0.1], num_examples=10)),
                    neg_child=LeafNode(
                        value=ProbabilityValue(
                            probability=[0.1, 0.8, 0.1], num_examples=20))),
                neg_child=LeafNode(
                    value=ProbabilityValue(
                        probability=[0.1, 0.1, 0.8], num_examples=30)))))

    builder.close()

    logging.info("Loading model")
    loaded_model = tf.keras.models.load_model(model_path)

    logging.info("Make predictions")
    tf_dataset = tf.data.Dataset.from_tensor_slices({
        "f1": [1.0, 2.0, 3.0],
        "f2": ["cat", "cat", "bird"]
    }).batch(2)
    predictions = loaded_model.predict(tf_dataset)
    self.assertAllClose(predictions,
                        [[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])

  def test_regression_random_forest(self):
    model_path = os.path.join(tmp_path(), "regression_rf")
    logging.info("Create model in %s", model_path)
    builder = builder_lib.RandomForestBuilder(
        path=model_path,
        model_format=builder_lib.ModelFormat.TENSORFLOW_SAVED_MODEL,
        objective=py_tree.objective.RegressionObjective(label="age"))

    #  f1>=1.5
    #    ├─(pos)─ age: 1
    #    └─(neg)─ age: 2
    builder.add_tree(
        Tree(
            NonLeafNode(
                condition=NumericalHigherThanCondition(
                    feature=SimpleColumnSpec(
                        name="f1", type=py_tree.dataspec.ColumnType.NUMERICAL),
                    threshold=1.5,
                    missing_evaluation=False),
                pos_child=LeafNode(
                    value=RegressionValue(value=1, num_examples=30)),
                neg_child=LeafNode(
                    value=RegressionValue(value=2, num_examples=30)))))

    builder.close()

    logging.info("Loading model")
    loaded_model = tf.keras.models.load_model(model_path)

    logging.info("Make predictions")
    tf_dataset = tf.data.Dataset.from_tensor_slices({
        "f1": [1.0, 2.0],
    }).batch(2)
    predictions = loaded_model.predict(tf_dataset)
    self.assertAllClose(predictions, [[2.0], [1.0]])

  def test_binary_classification_gbt(self):
    model_path = os.path.join(tmp_path(), "binary_classification_gbt")
    logging.info("Create model in %s", model_path)
    builder = builder_lib.GradientBoostedTreeBuilder(
        path=model_path,
        model_format=builder_lib.ModelFormat.TENSORFLOW_SAVED_MODEL,
        bias=1.0,
        objective=py_tree.objective.ClassificationObjective(
            label="color", classes=["red", "blue"]))

    #  bias: 1.0 (toward "blue")
    #  f1>=1.5
    #    ├─(pos)─ +1.0 (toward "blue")
    #    └─(neg)─ -1.0 (toward "blue")
    builder.add_tree(
        Tree(
            NonLeafNode(
                condition=NumericalHigherThanCondition(
                    feature=SimpleColumnSpec(
                        name="f1", type=py_tree.dataspec.ColumnType.NUMERICAL),
                    threshold=1.5,
                    missing_evaluation=False),
                pos_child=LeafNode(
                    value=RegressionValue(value=+1, num_examples=30)),
                neg_child=LeafNode(
                    value=RegressionValue(value=-1, num_examples=30)))))

    builder.close()

    logging.info("Loading model")
    loaded_model = tf.keras.models.load_model(model_path)

    logging.info("Make predictions")
    tf_dataset = tf.data.Dataset.from_tensor_slices({
        "f1": [1.0, 2.0],
    }).batch(2)
    predictions = loaded_model.predict(tf_dataset)
    self.assertAllClose(
        predictions,
        [[1.0 / (1.0 + math.exp(0.0))], [1.0 / (1.0 + math.exp(-2.0))]])

  def test_multi_class_classification_gbt(self):
    model_path = os.path.join(tmp_path(), "multi_class_classification_gbt")
    logging.info("Create model in %s", model_path)
    builder = builder_lib.GradientBoostedTreeBuilder(
        path=model_path,
        model_format=builder_lib.ModelFormat.TENSORFLOW_SAVED_MODEL,
        objective=py_tree.objective.ClassificationObjective(
            label="color", classes=["red", "blue", "green"]))

    #  f1>=1.5
    #    ├─(pos)─ +1.0 (toward "red")
    #    └─(neg)─ -1.0 (toward "red")
    #  f1>=2.5
    #    ├─(pos)─ +1.0 (toward "blue")
    #    └─(neg)─ -1.0 (toward "blue")
    #  f1>=3.5
    #    ├─(pos)─ +1.0 (toward "green")
    #    └─(neg)─ -1.0 (toward "green")

    for threshold in [1.5, 2.5, 3.5]:
      builder.add_tree(
          Tree(
              NonLeafNode(
                  condition=NumericalHigherThanCondition(
                      feature=SimpleColumnSpec(
                          name="f1",
                          type=py_tree.dataspec.ColumnType.NUMERICAL),
                      threshold=threshold,
                      missing_evaluation=False),
                  pos_child=LeafNode(
                      value=RegressionValue(value=+1, num_examples=30)),
                  neg_child=LeafNode(
                      value=RegressionValue(value=-1, num_examples=30)))))

    builder.close()

    logging.info("Loading model")
    loaded_model = tf.keras.models.load_model(model_path)

    logging.info("Make predictions")
    tf_dataset = tf.data.Dataset.from_tensor_slices({
        "f1": [1.0, 2.0],
    }).batch(2)
    predictions = loaded_model.predict(tf_dataset)

    soft_max_sum = np.sum(np.exp([+1, -1, -1]))
    self.assertAllClose(predictions, [[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                                      [
                                          math.exp(+1) / soft_max_sum,
                                          math.exp(-1) / soft_max_sum,
                                          math.exp(-1) / soft_max_sum
                                      ]])

  def test_regression_gbt(self):
    model_path = os.path.join(tmp_path(), "regression_gbt")
    logging.info("Create model in %s", model_path)
    builder = builder_lib.GradientBoostedTreeBuilder(
        path=model_path,
        bias=1.0,
        model_format=builder_lib.ModelFormat.TENSORFLOW_SAVED_MODEL,
        objective=py_tree.objective.RegressionObjective(label="age"))

    # bias: 1.0
    #  f1>=1.5
    #    ├─(pos)─ +1
    #    └─(neg)─ -1
    builder.add_tree(
        Tree(
            NonLeafNode(
                condition=NumericalHigherThanCondition(
                    feature=SimpleColumnSpec(
                        name="f1", type=py_tree.dataspec.ColumnType.NUMERICAL),
                    threshold=1.5,
                    missing_evaluation=False),
                pos_child=LeafNode(
                    value=RegressionValue(value=+1, num_examples=30)),
                neg_child=LeafNode(
                    value=RegressionValue(value=-1, num_examples=30)))))

    builder.close()

    logging.info("Loading model")
    loaded_model = tf.keras.models.load_model(model_path)

    logging.info("Make predictions")
    tf_dataset = tf.data.Dataset.from_tensor_slices({
        "f1": [1.0, 2.0],
    }).batch(2)
    predictions = loaded_model.predict(tf_dataset)
    self.assertAllClose(predictions, [[0.0], [2.0]])

  def test_ranking_gbt(self):
    model_path = os.path.join(tmp_path(), "ranking_gbt")
    logging.info("Create model in %s", model_path)
    builder = builder_lib.GradientBoostedTreeBuilder(
        path=model_path,
        bias=1.0,
        model_format=builder_lib.ModelFormat.TENSORFLOW_SAVED_MODEL,
        objective=py_tree.objective.RankingObjective(
            label="document", group="query"))

    # bias: 1.0
    #  f1>=1.5
    #    ├─(pos)─ +1
    #    └─(neg)─ -1
    builder.add_tree(
        Tree(
            NonLeafNode(
                condition=NumericalHigherThanCondition(
                    feature=SimpleColumnSpec(
                        name="f1", type=py_tree.dataspec.ColumnType.NUMERICAL),
                    threshold=1.5,
                    missing_evaluation=False),
                pos_child=LeafNode(
                    value=RegressionValue(value=+1, num_examples=30)),
                neg_child=LeafNode(
                    value=RegressionValue(value=-1, num_examples=30)))))

    builder.close()

    logging.info("Loading model")
    loaded_model = tf.keras.models.load_model(model_path)

    logging.info("Make predictions")
    tf_dataset = tf.data.Dataset.from_tensor_slices({
        "f1": [1.0, 2.0],
    }).batch(2)
    predictions = loaded_model.predict(tf_dataset)
    self.assertAllClose(predictions, [[0.0], [2.0]])

  def test_error_empty_path(self):
    self.assertRaises(
        ValueError, lambda: builder_lib.RandomForestBuilder(
            path="",
            model_format=builder_lib.ModelFormat.TENSORFLOW_SAVED_MODEL,
            objective=py_tree.objective.RegressionObjective("label")))

  def test_error_multi_tree_cart(self):
    builder = builder_lib.CARTBuilder(
        path=os.path.join(tmp_path(), "model"),
        objective=py_tree.objective.RegressionObjective("label"))
    builder.add_tree(Tree(LeafNode(RegressionValue(1, 30))))

    self.assertRaises(
        ValueError,
        lambda: builder.add_tree(Tree(LeafNode(RegressionValue(1, 30)))))

  def test_error_reg_cart_with_class_tree(self):
    builder = builder_lib.CARTBuilder(
        path=os.path.join(tmp_path(), "model"),
        objective=py_tree.objective.RegressionObjective("label"))
    self.assertRaises(
        ValueError, lambda: builder.add_tree(
            Tree(
                LeafNode(
                    ProbabilityValue(
                        probability=[0.8, 0.1, 0.1], num_examples=10)))))

  def test_error_class_cart_with_reg_tree(self):
    builder = builder_lib.CARTBuilder(
        path=os.path.join(tmp_path(), "model"),
        objective=py_tree.objective.ClassificationObjective(
            "label", classes=["red", "blue"]))
    self.assertRaises(
        ValueError,
        lambda: builder.add_tree(Tree(LeafNode(RegressionValue(1, 10)))))

  def test_error_wrong_class_leaf_dim(self):
    builder = builder_lib.CARTBuilder(
        path=os.path.join(tmp_path(), "model"),
        objective=py_tree.objective.ClassificationObjective(
            "label", classes=["red", "blue"]))
    self.assertRaises(
        ValueError, lambda: builder.add_tree(
            Tree(
                LeafNode(
                    ProbabilityValue(
                        probability=[0.8, 0.1, 0.1], num_examples=10)))))

  def test_error_gbt_with_class_tree(self):
    builder = builder_lib.GradientBoostedTreeBuilder(
        path=os.path.join(tmp_path(), "model"),
        objective=py_tree.objective.ClassificationObjective(
            "label", classes=["red", "blue", "green"]))

    self.assertRaises(
        ValueError, lambda: builder.add_tree(
            Tree(
                LeafNode(
                    ProbabilityValue(
                        probability=[0.8, 0.1, 0.1], num_examples=10)))))

  def test_error_gbt_wrong_number_of_trees(self):
    builder = builder_lib.GradientBoostedTreeBuilder(
        path=os.path.join(tmp_path(), "model"),
        objective=py_tree.objective.ClassificationObjective(
            "label", classes=["red", "blue", "green"]))

    builder.add_tree(Tree(LeafNode(RegressionValue(1, num_examples=10))))
    self.assertRaises(ValueError, builder.close)


if __name__ == "__main__":
  tf.test.main()
