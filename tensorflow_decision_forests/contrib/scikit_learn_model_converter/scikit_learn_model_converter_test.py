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

"""Tests for scikit_learn_model_converter."""

from absl.testing import parameterized
import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn import tree
import tensorflow as tf

from tensorflow_decision_forests.contrib import scikit_learn_model_converter


class ScikitLearnModelConverterTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters((tree.DecisionTreeRegressor(random_state=42),),
                            (tree.ExtraTreeRegressor(random_state=42),))
  def test_convert_reproduces_regression_model(
      self,
      sklearn_tree,
  ):
    features, labels = datasets.make_regression(
        n_samples=100,
        n_features=10,
        random_state=42,
    )
    sklearn_tree.fit(features, labels)
    tf_tree = scikit_learn_model_converter.convert(sklearn_tree)
    tf_features = tf.constant(features, dtype=tf.float32)
    tf_labels = tf_tree(tf_features).numpy().ravel()
    sklearn_labels = sklearn_tree.predict(features).astype(np.float32)
    self.assertAllEqual(sklearn_labels, tf_labels)

  @parameterized.parameters((tree.DecisionTreeClassifier(random_state=42),),
                            (tree.ExtraTreeClassifier(random_state=42),))
  def test_convert_reproduces_classification_model(
      self,
      sklearn_tree,
  ):
    features, labels = datasets.make_classification(
        n_samples=100,
        n_features=10,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=42,
    )
    sklearn_tree.fit(features, labels)
    tf_tree = scikit_learn_model_converter.convert(sklearn_tree)
    tf_features = tf.constant(features, dtype=tf.float32)
    tf_labels = tf_tree(tf_features).numpy()
    sklearn_labels = sklearn_tree.predict_proba(features).astype(np.float32)
    self.assertAllEqual(sklearn_labels, tf_labels)

  def test_convert_raises_when_unrecognised_model_provided(self):
    features, labels = datasets.make_regression(
        n_samples=100,
        n_features=10,
        random_state=42,
    )
    sklearn_model = linear_model.LinearRegression().fit(features, labels)
    with self.assertRaises(NotImplementedError):
      scikit_learn_model_converter.convert(sklearn_model)

  def test_convert_raises_when_sklearn_model_is_not_fit(self):
    with self.assertRaises(
        ValueError,
        msg="Scikit-learn model must be fit to data before converting to TF.",
    ):
      _ = scikit_learn_model_converter.convert(tree.DecisionTreeRegressor())

  def test_convert_raises_when_regression_target_is_multivariate(self):
    features, labels = datasets.make_regression(
        n_samples=100,
        n_features=10,
        # This produces a two-dimensional target variable.
        n_targets=2,
        random_state=42,
    )
    sklearn_tree = tree.DecisionTreeRegressor().fit(features, labels)
    with self.assertRaisesRegex(
        ValueError,
        "Only scalar regression and single-label classification are supported.",
    ):
      _ = scikit_learn_model_converter.convert(sklearn_tree)

  def test_convert_raises_when_classification_target_is_multilabel(self):
    features, labels = datasets.make_multilabel_classification(
        n_samples=100,
        n_features=10,
        # This assigns two class labels per example.
        n_labels=2,
        random_state=42,
    )
    sklearn_tree = tree.DecisionTreeClassifier().fit(features, labels)
    with self.assertRaisesRegex(
        ValueError,
        "Only scalar regression and single-label classification are supported.",
    ):
      _ = scikit_learn_model_converter.convert(sklearn_tree)

  def test_convert_uses_intermediate_model_path_if_provided(self):
    features, labels = datasets.make_classification(
        n_samples=100,
        n_features=10,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=42,
    )
    sklearn_tree = tree.DecisionTreeClassifier().fit(features, labels)
    write_path = self.create_tempdir()
    _ = scikit_learn_model_converter.convert(
        sklearn_tree,
        intermediate_write_path=write_path,
    )
    # We should be able to load the intermediate TFDF model from the given path.
    tfdf_tree = tf.keras.models.load_model(write_path)
    self.assertIsInstance(tfdf_tree, tf.keras.Model)


if __name__ == "__main__":
  tf.test.main()
