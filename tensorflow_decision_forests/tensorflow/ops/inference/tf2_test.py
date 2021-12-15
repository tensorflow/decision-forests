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

import concurrent.futures
import os
import tempfile

from absl import logging
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_decision_forests.tensorflow.ops.inference import api as inference
from tensorflow_decision_forests.tensorflow.ops.inference import test_utils
from absl import flags


def data_root_path() -> str:
  return ""


def test_data_path() -> str:
  return os.path.join(data_root_path(),
                      "external/ydf/yggdrasil_decision_forests/test_data")


class TfOpTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ("base", False, False),
      ("boolean", True, False),
      ("catset", False, True),
  )
  def test_toy_rf_classification_winner_takes_all(self, add_boolean_features,
                                                  has_catset):

    # Create toy model.
    model_path = os.path.join(
        tempfile.mkdtemp(dir=self.get_temp_dir()), "test_basic_rf_wta")
    test_utils.build_toy_random_forest(
        model_path,
        winner_take_all_inference=True,
        add_boolean_features=add_boolean_features,
        has_catset=has_catset)
    features = test_utils.build_toy_input_feature_values(
        features=None, has_catset=has_catset)

    # Prepare model.
    model = inference.Model(model_path)

    @tf.function
    def init_model():
      tf.print("Loading model")
      model.init_op()

    @tf.function
    def apply_model(features):
      tf.print("Running model")
      return model.apply(features)

    init_model()

    predictions = apply_model(features)
    print("Predictions: %s", predictions)

    logging.info("dense_predictions_values: %s", predictions.dense_predictions)
    logging.info("dense_col_representation_values: %s",
                 predictions.dense_col_representation)

    expected_proba, expected_classes = test_utils.expected_toy_predictions_rf_wta(
        add_boolean_features=add_boolean_features, has_catset=has_catset)
    self.assertAllEqual(predictions.dense_col_representation, expected_classes)
    self.assertAllClose(predictions.dense_predictions, expected_proba)

  @parameterized.named_parameters(
      ("base", False, False),
      ("boolean", True, False),
      ("catset", False, True),
  )
  def test_toy_rf_classification_winner_takes_all_v2(self, add_boolean_features,
                                                     has_catset):

    # Create toy model.
    model_path = os.path.join(
        tempfile.mkdtemp(dir=self.get_temp_dir()), "test_basic_rf_wta")
    test_utils.build_toy_random_forest(
        model_path,
        winner_take_all_inference=True,
        add_boolean_features=add_boolean_features,
        has_catset=has_catset)
    features = test_utils.build_toy_input_feature_values(
        features=None, has_catset=has_catset)

    # Prepare model.
    tf.print("Loading model")
    model = inference.ModelV2(model_path)

    @tf.function
    def apply_non_eager(x):
      return model.apply(x)

    predictions_non_eager = apply_non_eager(features)
    predictions_eager = model.apply(features)

    def check_predictions(predictions):
      print("Predictions: %s", predictions)

      logging.info("dense_predictions_values: %s",
                   predictions.dense_predictions)
      logging.info("dense_col_representation_values: %s",
                   predictions.dense_col_representation)

      (expected_proba,
       expected_classes) = test_utils.expected_toy_predictions_rf_wta(
           add_boolean_features=add_boolean_features, has_catset=has_catset)
      self.assertAllEqual(predictions.dense_col_representation,
                          expected_classes)
      self.assertAllClose(predictions.dense_predictions, expected_proba)

    check_predictions(predictions_non_eager)
    check_predictions(predictions_eager)

  def test_multi_thread(self):

    # Create toy model with a lot of trees i.e. slow model.
    model_path = os.path.join(
        tempfile.mkdtemp(dir=self.get_temp_dir()), "test_multi_thread")
    test_utils.build_toy_random_forest(
        model_path, winner_take_all_inference=True, num_trees=100000)
    features = test_utils.build_toy_input_feature_values(features=None)

    model = inference.ModelV2(model_path)

    @tf.function
    def apply_non_eager(x):
      return model.apply(x)

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
      predictions = executor.map(apply_non_eager, [features] * 1000)

    (expected_proba,
     expected_classes) = test_utils.expected_toy_predictions_rf_wta()

    for prediction in predictions:
      self.assertAllEqual(prediction.dense_col_representation, expected_classes)
      self.assertAllClose(prediction.dense_predictions, expected_proba)

  def test_get_leaves(self):
    """Access the active leaves of the model."""

    model_path = os.path.join(
        tempfile.mkdtemp(dir=self.get_temp_dir()), "test_get_leaves")
    test_utils.build_toy_random_forest(
        model_path, winner_take_all_inference=True, num_trees=6)
    features = test_utils.build_toy_input_feature_values(features=None)

    model = inference.ModelV2(model_path, output_types=["LEAVES"])

    leaves = model.apply_get_leaves(features)
    logging.info("Leaves: %s", leaves)

    self.assertAllEqual(leaves, [[3] * 6, [2] * 6, [1] * 6, [0] * 6])

  def test_get_leaves_real_rf(self):

    model_path = os.path.join(test_data_path(), "model",
                              "adult_binary_class_rf")

    model = inference.ModelV2(model_path, output_types=["LEAVES"])

    def f(x):
      return tf.constant(x)

    # First two examples of adult_test.csv
    features = {
        "age": f([39.0, 50.0]),
        "workclass": f(["State-gov", "Self-emp-not-inc"]),
        "fnlwgt": f([77516.0, 83311.0]),
        "education": f(["Bachelors", "Bachelors"]),
        "education_num": f([13, 13]),
        "marital_status": f(["Never-married", "Married-civ-spouse"]),
        "occupation": f(["Adm-clerical", "Exec-managerial"]),
        "relationship": f(["Not-in-family", "Husband"]),
        "race": f(["White", "White"]),
        "sex": f(["Male", "Male"]),
        "capital_gain": f([2174.0, 0.0]),
        "capital_loss": f([0.0, 0.0]),
        "hours_per_week": f([40.0, 13.0]),
        "native_country": f(["United-States", "United-States"]),
    }

    leaves = model.apply_get_leaves(features)

    self.assertEqual(leaves.shape, (2, 100))
    self.assertAllEqual(leaves[0, :7], [156, 119, 139, 319, 215, 50, 151])




if __name__ == "__main__":
  tf.test.main()
