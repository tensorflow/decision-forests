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
import pandas as pd
import tensorflow as tf
import tf_keras

from tensorflow_decision_forests import keras
from tensorflow_decision_forests.component import py_tree
from tensorflow_decision_forests.component.builder import builder as builder_lib
from tensorflow_decision_forests.component.inspector import inspector as inspector_lib

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
  return os.path.join(
      data_root_path(), "external/ydf/yggdrasil_decision_forests/test_data"
  )


def tmp_path() -> str:
  return flags.FLAGS.test_tmpdir


def test_model_directory() -> str:
  return os.path.join(test_data_path(), "model")


def test_dataset_directory() -> str:
  return os.path.join(test_data_path(), "dataset")


class BuilderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (None, None), ("", "abc123"), ("prefix_", "test_model")
  )
  def test_classification_random_forest(self, file_prefix, model_name):
    model_path = os.path.join(tmp_path(), "classification_rf")
    logging.info("Create model in %s", model_path)
    builder = builder_lib.RandomForestBuilder(
        path=model_path,
        model_format=builder_lib.ModelFormat.TENSORFLOW_SAVED_MODEL,
        objective=py_tree.objective.ClassificationObjective(
            label="color", classes=["red", "blue", "green"]
        ),
        file_prefix=file_prefix,
        keras_model_name=model_name,
    )

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
                        name="f1", type=py_tree.dataspec.ColumnType.NUMERICAL
                    ),
                    threshold=1.5,
                    missing_evaluation=False,
                ),
                pos_child=NonLeafNode(
                    condition=CategoricalIsInCondition(
                        feature=SimpleColumnSpec(
                            name="f2",
                            type=py_tree.dataspec.ColumnType.CATEGORICAL,
                        ),
                        mask=["cat", "dog"],
                        missing_evaluation=False,
                    ),
                    pos_child=LeafNode(
                        value=ProbabilityValue(
                            probability=[0.8, 0.1, 0.1], num_examples=10
                        )
                    ),
                    neg_child=LeafNode(
                        value=ProbabilityValue(
                            probability=[0.1, 0.8, 0.1], num_examples=20
                        )
                    ),
                ),
                neg_child=LeafNode(
                    value=ProbabilityValue(
                        probability=[0.1, 0.1, 0.8], num_examples=30
                    )
                ),
            )
        )
    )

    builder.close()

    if file_prefix is not None:
      self.assertEqual(
          inspector_lib.detect_model_file_prefix(
              os.path.join(model_path, "assets")
          ),
          file_prefix,
      )

    logging.info("Loading model")
    loaded_model = tf_keras.models.load_model(model_path)
    expected_model_name = (
        "inference_core_model" if model_name is None else model_name
    )
    self.assertEqual(loaded_model.name, expected_model_name)
    logging.info("Make predictions")
    tf_dataset = tf.data.Dataset.from_tensor_slices(
        {"f1": [1.0, 2.0, 3.0], "f2": ["cat", "cat", "bird"]}
    ).batch(2)
    predictions = loaded_model.predict(tf_dataset)
    self.assertAllClose(
        predictions, [[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]
    )

  @parameterized.parameters((None,), ("",), ("prefix_",))
  def test_classification_cart(self, file_prefix):
    model_path = os.path.join(tmp_path(), "classification_cart")
    logging.info("Create model in %s", model_path)
    builder = builder_lib.CARTBuilder(
        path=model_path,
        model_format=builder_lib.ModelFormat.TENSORFLOW_SAVED_MODEL,
        objective=py_tree.objective.ClassificationObjective(
            label="color", classes=["red", "blue", "green"]
        ),
        file_prefix=file_prefix,
        keras_model_name="classification_cart",
    )

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
                        name="f1", type=py_tree.dataspec.ColumnType.NUMERICAL
                    ),
                    threshold=1.5,
                    missing_evaluation=False,
                ),
                pos_child=NonLeafNode(
                    condition=CategoricalIsInCondition(
                        feature=SimpleColumnSpec(
                            name="f2",
                            type=py_tree.dataspec.ColumnType.CATEGORICAL,
                        ),
                        mask=["cat", "dog"],
                        missing_evaluation=False,
                    ),
                    pos_child=LeafNode(
                        value=ProbabilityValue(
                            probability=[0.8, 0.1, 0.1], num_examples=10
                        )
                    ),
                    neg_child=LeafNode(
                        value=ProbabilityValue(
                            probability=[0.1, 0.8, 0.1], num_examples=20
                        )
                    ),
                ),
                neg_child=LeafNode(
                    value=ProbabilityValue(
                        probability=[0.1, 0.1, 0.8], num_examples=30
                    )
                ),
            )
        )
    )

    builder.close()

    if file_prefix is not None:
      self.assertEqual(
          inspector_lib.detect_model_file_prefix(
              os.path.join(model_path, "assets")
          ),
          file_prefix,
      )

    logging.info("Loading model")
    loaded_model = tf_keras.models.load_model(model_path)
    self.assertEqual(loaded_model.name, "classification_cart")
    logging.info("Make predictions")
    tf_dataset = tf.data.Dataset.from_tensor_slices(
        {"f1": [1.0, 2.0, 3.0], "f2": ["cat", "cat", "bird"]}
    ).batch(2)
    predictions = loaded_model.predict(tf_dataset)
    self.assertAllClose(
        predictions, [[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]
    )

  def test_regression_random_forest(self):
    model_path = os.path.join(tmp_path(), "regression_rf")
    logging.info("Create model in %s", model_path)
    builder = builder_lib.RandomForestBuilder(
        path=model_path,
        model_format=builder_lib.ModelFormat.TENSORFLOW_SAVED_MODEL,
        objective=py_tree.objective.RegressionObjective(label="age"),
    )

    #  f1>=1.5
    #    ├─(pos)─ age: 1
    #    └─(neg)─ age: 2
    builder.add_tree(
        Tree(
            NonLeafNode(
                condition=NumericalHigherThanCondition(
                    feature=SimpleColumnSpec(
                        name="f1", type=py_tree.dataspec.ColumnType.NUMERICAL
                    ),
                    threshold=1.5,
                    missing_evaluation=False,
                ),
                pos_child=LeafNode(
                    value=RegressionValue(value=1, num_examples=30)
                ),
                neg_child=LeafNode(
                    value=RegressionValue(value=2, num_examples=30)
                ),
            )
        )
    )

    builder.close()

    logging.info("Loading model")
    loaded_model = tf_keras.models.load_model(model_path)

    logging.info("Make predictions")
    tf_dataset = tf.data.Dataset.from_tensor_slices({
        "f1": [1.0, 2.0],
    }).batch(2)
    predictions = loaded_model.predict(tf_dataset)
    self.assertAllClose(predictions, [[2.0], [1.0]])

  def test_regression_random_forest_with_categorical_integer(self):
    model_path = os.path.join(tmp_path(), "regression_rf_with_cat_int")
    logging.info("Create model in %s", model_path)
    builder = builder_lib.RandomForestBuilder(
        path=model_path,
        model_format=builder_lib.ModelFormat.TENSORFLOW_SAVED_MODEL,
        objective=py_tree.objective.RegressionObjective(label="age"),
        advanced_arguments=builder_lib.AdvancedArguments(
            disable_categorical_integer_offset_correction=True
        ),
    )

    #  f1 in [2,3]
    #    ├─(pos)─ age: 1
    #    └─(neg)─ age: 2
    builder.add_tree(
        Tree(
            NonLeafNode(
                condition=CategoricalIsInCondition(
                    feature=SimpleColumnSpec(
                        name="f1", type=py_tree.dataspec.ColumnType.CATEGORICAL
                    ),
                    mask=[2, 3],
                    missing_evaluation=False,
                ),
                pos_child=LeafNode(
                    value=RegressionValue(value=1, num_examples=30)
                ),
                neg_child=LeafNode(
                    value=RegressionValue(value=2, num_examples=30)
                ),
            )
        )
    )

    builder.close()

    logging.info("Loading model")
    loaded_model = tf_keras.models.load_model(model_path)

    logging.info("Make predictions")
    tf_dataset = tf.data.Dataset.from_tensor_slices({
        "f1": [1, 2, 3, 4],
    }).batch(2)
    predictions = loaded_model.predict(tf_dataset)
    self.assertAllClose(predictions, [[2], [1], [1], [2]])

  def test_binary_classification_gbt(self):
    model_path = os.path.join(tmp_path(), "binary_classification_gbt")
    logging.info("Create model in %s", model_path)
    builder = builder_lib.GradientBoostedTreeBuilder(
        path=model_path,
        model_format=builder_lib.ModelFormat.TENSORFLOW_SAVED_MODEL,
        bias=1.0,
        objective=py_tree.objective.ClassificationObjective(
            label="color", classes=["red", "blue"]
        ),
        keras_model_name="binary_classification_gbt",
    )

    #  bias: 1.0 (toward "blue")
    #  f1>=1.5
    #    ├─(pos)─ +1.0 (toward "blue")
    #    └─(neg)─ -1.0 (toward "blue")
    builder.add_tree(
        Tree(
            NonLeafNode(
                condition=NumericalHigherThanCondition(
                    feature=SimpleColumnSpec(
                        name="f1", type=py_tree.dataspec.ColumnType.NUMERICAL
                    ),
                    threshold=1.5,
                    missing_evaluation=False,
                ),
                pos_child=LeafNode(
                    value=RegressionValue(value=+1, num_examples=30)
                ),
                neg_child=LeafNode(
                    value=RegressionValue(value=-1, num_examples=30)
                ),
            )
        )
    )

    builder.close()

    logging.info("Loading model")
    loaded_model = tf_keras.models.load_model(model_path)
    self.assertEqual(loaded_model.name, "binary_classification_gbt")
    logging.info("Make predictions")
    tf_dataset = tf.data.Dataset.from_tensor_slices({
        "f1": [1.0, 2.0],
    }).batch(2)
    predictions = loaded_model.predict(tf_dataset)
    self.assertAllClose(
        predictions,
        [[1.0 / (1.0 + math.exp(0.0))], [1.0 / (1.0 + math.exp(-2.0))]],
    )

  @parameterized.parameters((None,), ("",), ("prefix_",))
  def test_multi_class_classification_gbt(self, file_prefix):
    model_path = os.path.join(tmp_path(), "multi_class_classification_gbt")
    logging.info("Create model in %s", model_path)
    builder = builder_lib.GradientBoostedTreeBuilder(
        path=model_path,
        model_format=builder_lib.ModelFormat.TENSORFLOW_SAVED_MODEL,
        objective=py_tree.objective.ClassificationObjective(
            label="color", classes=["red", "blue", "green"]
        ),
        file_prefix=file_prefix,
    )

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
                          name="f1", type=py_tree.dataspec.ColumnType.NUMERICAL
                      ),
                      threshold=threshold,
                      missing_evaluation=False,
                  ),
                  pos_child=LeafNode(
                      value=RegressionValue(value=+1, num_examples=30)
                  ),
                  neg_child=LeafNode(
                      value=RegressionValue(value=-1, num_examples=30)
                  ),
              )
          )
      )

    builder.close()

    if file_prefix is not None:
      self.assertEqual(
          inspector_lib.detect_model_file_prefix(
              os.path.join(model_path, "assets")
          ),
          file_prefix,
      )

    logging.info("Loading model")
    loaded_model = tf_keras.models.load_model(model_path)

    logging.info("Make predictions")
    tf_dataset = tf.data.Dataset.from_tensor_slices({
        "f1": [1.0, 2.0],
    }).batch(2)
    predictions = loaded_model.predict(tf_dataset)

    soft_max_sum = np.sum(np.exp([+1, -1, -1]))
    self.assertAllClose(
        predictions,
        [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [
                math.exp(+1) / soft_max_sum,
                math.exp(-1) / soft_max_sum,
                math.exp(-1) / soft_max_sum,
            ],
        ],
    )

  def test_regression_gbt(self):
    model_path = os.path.join(tmp_path(), "regression_gbt")
    logging.info("Create model in %s", model_path)
    builder = builder_lib.GradientBoostedTreeBuilder(
        path=model_path,
        bias=1.0,
        model_format=builder_lib.ModelFormat.TENSORFLOW_SAVED_MODEL,
        objective=py_tree.objective.RegressionObjective(label="age"),
    )

    # bias: 1.0
    #  f1>=1.5
    #    ├─(pos)─ +1
    #    └─(neg)─ -1
    builder.add_tree(
        Tree(
            NonLeafNode(
                condition=NumericalHigherThanCondition(
                    feature=SimpleColumnSpec(
                        name="f1", type=py_tree.dataspec.ColumnType.NUMERICAL
                    ),
                    threshold=1.5,
                    missing_evaluation=False,
                ),
                pos_child=LeafNode(
                    value=RegressionValue(value=+1, num_examples=30)
                ),
                neg_child=LeafNode(
                    value=RegressionValue(value=-1, num_examples=30)
                ),
            )
        )
    )

    builder.close()

    logging.info("Loading model")
    loaded_model = tf_keras.models.load_model(model_path)

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
            label="document", group="query"
        ),
    )

    # bias: 1.0
    #  f1>=1.5
    #    ├─(pos)─ +1
    #    └─(neg)─ -1
    builder.add_tree(
        Tree(
            NonLeafNode(
                condition=NumericalHigherThanCondition(
                    feature=SimpleColumnSpec(
                        name="f1", type=py_tree.dataspec.ColumnType.NUMERICAL
                    ),
                    threshold=1.5,
                    missing_evaluation=False,
                ),
                pos_child=LeafNode(
                    value=RegressionValue(value=+1, num_examples=30)
                ),
                neg_child=LeafNode(
                    value=RegressionValue(value=-1, num_examples=30)
                ),
            )
        )
    )

    builder.close()

    logging.info("Loading model")
    loaded_model = tf_keras.models.load_model(model_path)

    logging.info("Make predictions")
    tf_dataset = tf.data.Dataset.from_tensor_slices({
        "f1": [1.0, 2.0],
    }).batch(2)
    predictions = loaded_model.predict(tf_dataset)
    self.assertAllClose(predictions, [[0.0], [2.0]])

  def test_error_empty_path(self):
    self.assertRaises(
        ValueError,
        lambda: builder_lib.RandomForestBuilder(
            path="",
            model_format=builder_lib.ModelFormat.TENSORFLOW_SAVED_MODEL,
            objective=py_tree.objective.RegressionObjective("label"),
        ),
    )

  def test_error_multi_tree_cart(self):
    builder = builder_lib.CARTBuilder(
        path=os.path.join(tmp_path(), "model"),
        objective=py_tree.objective.RegressionObjective("label"),
    )
    builder.add_tree(Tree(LeafNode(RegressionValue(1, 30))))

    self.assertRaises(
        ValueError,
        lambda: builder.add_tree(Tree(LeafNode(RegressionValue(1, 30)))),
    )

  def test_error_reg_cart_with_class_tree(self):
    builder = builder_lib.CARTBuilder(
        path=os.path.join(tmp_path(), "model"),
        objective=py_tree.objective.RegressionObjective("label"),
    )
    self.assertRaises(
        ValueError,
        lambda: builder.add_tree(
            Tree(
                LeafNode(
                    ProbabilityValue(
                        probability=[0.8, 0.1, 0.1], num_examples=10
                    )
                )
            )
        ),
    )

  def test_error_class_cart_with_reg_tree(self):
    builder = builder_lib.CARTBuilder(
        path=os.path.join(tmp_path(), "model"),
        objective=py_tree.objective.ClassificationObjective(
            "label", classes=["red", "blue"]
        ),
    )
    self.assertRaises(
        ValueError,
        lambda: builder.add_tree(Tree(LeafNode(RegressionValue(1, 10)))),
    )

  def test_error_wrong_class_leaf_dim(self):
    builder = builder_lib.CARTBuilder(
        path=os.path.join(tmp_path(), "model"),
        objective=py_tree.objective.ClassificationObjective(
            "label", classes=["red", "blue"]
        ),
    )
    self.assertRaises(
        ValueError,
        lambda: builder.add_tree(
            Tree(
                LeafNode(
                    ProbabilityValue(
                        probability=[0.8, 0.1, 0.1], num_examples=10
                    )
                )
            )
        ),
    )

  def test_error_gbt_with_class_tree(self):
    builder = builder_lib.GradientBoostedTreeBuilder(
        path=os.path.join(tmp_path(), "model"),
        objective=py_tree.objective.ClassificationObjective(
            "label", classes=["red", "blue", "green"]
        ),
    )

    self.assertRaises(
        ValueError,
        lambda: builder.add_tree(
            Tree(
                LeafNode(
                    ProbabilityValue(
                        probability=[0.8, 0.1, 0.1], num_examples=10
                    )
                )
            )
        ),
    )

  def test_error_gbt_wrong_number_of_trees(self):
    builder = builder_lib.GradientBoostedTreeBuilder(
        path=os.path.join(tmp_path(), "model"),
        objective=py_tree.objective.ClassificationObjective(
            "label", classes=["red", "blue", "green"]
        ),
    )

    builder.add_tree(Tree(LeafNode(RegressionValue(1, num_examples=10))))
    self.assertRaises(ValueError, builder.close)

  def test_get_set_dictionary(self):
    builder = builder_lib.RandomForestBuilder(
        path=os.path.join(tmp_path(), "model"),
        objective=py_tree.objective.ClassificationObjective(
            "label", classes=["true", "false"]
        ),
    )

    builder.add_tree(
        Tree(
            NonLeafNode(
                condition=CategoricalIsInCondition(
                    feature=SimpleColumnSpec(
                        name="f1", type=py_tree.dataspec.ColumnType.CATEGORICAL
                    ),
                    mask=["x", "y"],
                    missing_evaluation=False,
                ),
                pos_child=LeafNode(
                    value=ProbabilityValue(
                        probability=[0.8, 0.2], num_examples=10
                    )
                ),
                neg_child=LeafNode(
                    value=ProbabilityValue(
                        probability=[0.2, 0.8], num_examples=20
                    )
                ),
            )
        )
    )

    self.assertEqual(builder.get_dictionary("f1"), ["<OOD>", "x", "y"])
    builder.set_dictionary("f1", ["<OOD>", "x", "y", "z"])
    self.assertEqual(builder.get_dictionary("f1"), ["<OOD>", "x", "y", "z"])
    builder.close()

  def test_extract_random_forest(self):
    """Extract 5 trees from a trained RF model, and pack them into a model."""

    # Load a dataset
    dataset_path = os.path.join(test_dataset_directory(), "adult_test.csv")
    dataframe = pd.read_csv(dataset_path)
    # This "adult_binary_class_rf" model expect for "education_num" to be a
    # string.
    dataframe["education_num"] = dataframe["education_num"].astype(str)
    dataset = keras.pd_dataframe_to_tf_dataset(dataframe, "income")

    # Load an inspector to an existing model.
    src_model_path = os.path.join(
        test_model_directory(), "adult_binary_class_rf"
    )
    inspector = inspector_lib.make_inspector(src_model_path)

    # Extract a piece of this model
    def custom_model_input_signature(
        inspector: inspector_lib.AbstractInspector,
    ):
      input_spec = keras.build_default_input_model_signature(inspector)
      # Those features are stored as int64 in the dataset.
      for feature_name in [
          "age",
          "fnlwgt",
          "capital_gain",
          "capital_loss",
          "hours_per_week",
      ]:
        input_spec[feature_name] = tf.TensorSpec(shape=[None], dtype=tf.int64)
      return input_spec

    dst_model_path = os.path.join(tmp_path(), "model")
    builder = builder_lib.RandomForestBuilder(
        path=dst_model_path,
        objective=inspector.objective(),
        # Make sure the features and feature dictionaries are the same as in the
        # original model.
        import_dataspec=inspector.dataspec,
        input_signature_example_fn=custom_model_input_signature,
    )

    # Extract the first 5 trees
    for i in range(5):
      tree = inspector.extract_tree(i)
      builder.add_tree(tree)

    builder.close()

    truncated_model = tf_keras.models.load_model(dst_model_path)
    predictions = truncated_model.predict(dataset)
    self.assertEqual(predictions.shape, (9769, 1))

  def test_fast_serving_with_custom_numerical_default_evaluation(self):
    model_path = os.path.join(tmp_path(), "regression_gbt")
    logging.info("Create model in %s", model_path)
    builder = builder_lib.GradientBoostedTreeBuilder(
        path=model_path,
        bias=0.0,
        model_format=builder_lib.ModelFormat.TENSORFLOW_SAVED_MODEL,
        objective=py_tree.objective.RegressionObjective(label="label"),
    )

    # f1>=-1.0 (default: false)
    #   │
    #   ├─f1>=2.0 (default: false)
    #   │    │
    #   │    ├─1
    #   │    └─2
    #   └─f2>=-3.0 (default: true)
    #        │
    #        ├─f2>=4.0 (default: false)
    #        │    │
    #        │    ├─3
    #        │    └─4
    #        └─5

    def condition(feature, threshold, missing_evaluation, pos, neg):
      return NonLeafNode(
          condition=NumericalHigherThanCondition(
              feature=SimpleColumnSpec(
                  name=feature, type=py_tree.dataspec.ColumnType.NUMERICAL
              ),
              threshold=threshold,
              missing_evaluation=missing_evaluation,
          ),
          pos_child=pos,
          neg_child=neg,
      )

    def leaf(value):
      return LeafNode(RegressionValue(value=value, num_examples=1))

    builder.add_tree(
        Tree(
            condition(
                "f1",
                -1.0,
                False,
                condition("f1", 2.0, False, leaf(1), leaf(2)),
                condition(
                    "f2",
                    -3.0,
                    True,
                    condition("f2", 4.0, False, leaf(3), leaf(4)),
                    leaf(5),
                ),
            )
        )
    )
    builder.close()

    logging.info("Loading model")

    # There is no easy way to assert that an optimized inference engine was
    # chosen. If checking manually, make sure the "Use fast generic engine"
    # string is present (instead of the "Use slow generic engine" string).
    #
    # TODO:: Add API to check which inference engine is used.

    loaded_model = tf_keras.models.load_model(model_path)

    logging.info("Make predictions")
    tf_dataset = tf.data.Dataset.from_tensor_slices({
        "f1": [math.nan, 1.0, -2.0],
        "f2": [-4.0, -4.0, math.nan],
    }).batch(2)
    predictions = loaded_model.predict(tf_dataset)
    self.assertAllClose(predictions, [[5.0], [2.0], [4.0]])

    inspector = inspector_lib.make_inspector(os.path.join(model_path, "assets"))
    self.assertEqual(inspector.dataspec.columns[1].numerical.mean, -1.0 - 0.5)
    self.assertEqual(
        inspector.dataspec.columns[2].numerical.mean, (4.0 - 3.0) / 2.0
    )

  def test_categorical_is_in_global_imputation(self):
    model_path = os.path.join(tmp_path(), "categorical_is_in_global_imputation")
    builder = builder_lib.CARTBuilder(
        path=model_path,
        model_format=builder_lib.ModelFormat.TENSORFLOW_SAVED_MODEL,
        objective=py_tree.objective.RegressionObjective(label="color"),
        advanced_arguments=builder_lib.AdvancedArguments(
            disable_categorical_integer_offset_correction=True
        ),
    )

    #  f1 in [1, 2]
    #    ├─(pos)─ 1
    #    └─(neg)─ 2
    builder.add_tree(
        Tree(
            NonLeafNode(
                condition=CategoricalIsInCondition(
                    feature=SimpleColumnSpec(
                        name="f1", type=py_tree.dataspec.ColumnType.CATEGORICAL
                    ),
                    mask=[1, 2],
                    missing_evaluation=False,
                ),
                pos_child=LeafNode(value=RegressionValue(value=1)),
                neg_child=LeafNode(value=RegressionValue(value=2)),
            )
        )
    )

    builder.close()

    inspector = inspector_lib.make_inspector(os.path.join(model_path, "assets"))
    self.assertEqual(
        inspector.dataspec.columns[1].categorical.most_frequent_value, 3
    )
    self.assertEqual(
        inspector.dataspec.columns[1].categorical.number_of_unique_values, 4
    )


if __name__ == "__main__":
  tf.test.main()
