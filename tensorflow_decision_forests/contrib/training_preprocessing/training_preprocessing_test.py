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

"""Tests for training_preprocessing."""

import os

from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_decision_forests as tfdf

from tensorflow_decision_forests.contrib.training_preprocessing import training_preprocessing


class TrainingPreprocessingTest(tf.test.TestCase, parameterized.TestCase):

  def test_missing_labels_are_imputed(self):
    multitask_items = [
        tfdf.keras.MultiTaskItem(
            label='task_1', task=tfdf.keras.Task.CLASSIFICATION, primary=True
        ),
        tfdf.keras.MultiTaskItem(
            label='task_2', task=tfdf.keras.Task.REGRESSION, primary=False
        ),
    ]

    train_x = {'feature_1': tf.constant([1.0, 2.0, 3.0])}
    train_y = {
        'task_1': tf.constant([0, 1, 0]),
        'task_2': tf.constant([0.5, 0.0, 0.6]),
    }
    train_weights = {
        'task_1': tf.constant([1, 1, 0]),
        'task_2': tf.constant([1, 0, 1]),
    }

    result_x, result_y, result_weight = (
        training_preprocessing.multitask_label_presence_processing(
            multitask_items
        )(train_x, train_y, train_weights)
    )

    self.assertAllClose(result_x['feature_1'], train_x['feature_1'])
    self.assertAllClose(result_y['task_1'], tf.constant([0, 1, -2]))
    self.assertAllClose(
        result_y['task_2'], tf.constant([0.5, float('nan'), 0.6])
    )
    self.assertAllClose(result_weight, tf.constant([1, 1, 1]))

  @parameterized.named_parameters(
      dict(
          testcase_name='non_dict_weights',
          train_y={
              'task_1': tf.constant([0, 1, 0]),
              'task_2': tf.constant([0.5, 0.0, 0.6]),
          },
          train_weights=tf.constant([1, 1, 0]),
          expected_error=ValueError,
          expected_error_message='The preprocessor expects the',
      ),
      dict(
          testcase_name='non_dict_labels',
          train_y=tf.constant([0, 1, 0]),
          train_weights={
              'task_1': tf.constant([1, 1, 0]),
              'task_2': tf.constant([1, 0, 1]),
          },
          expected_error=ValueError,
          expected_error_message='The preprocessor expects the',
      ),
      dict(
          testcase_name='missing_task_in_labels',
          train_y={
              'task_1': tf.constant([0, 1, 0]),
              'task_unknown': tf.constant([0.5, 0.0, 0.6]),
          },
          train_weights={
              'task_1': tf.constant([1, 1, 0]),
              'task_2': tf.constant([1, 0, 1]),
          },
          expected_error=KeyError,
          expected_error_message='Task task_2 was not found in train_y',
      ),
      dict(
          testcase_name='missing_task_in_sample_weights',
          train_y={
              'task_1': tf.constant([0, 1, 0]),
              'task_2': tf.constant([0.5, 0.0, 0.6]),
          },
          train_weights={
              'task_unknown': tf.constant([1, 1, 0]),
              'task_2': tf.constant([1, 0, 1]),
          },
          expected_error=KeyError,
          expected_error_message='Task task_1 was not found in train_weights',
      ),
  )
  def test_rejected_setups(
      self, train_y, train_weights, expected_error, expected_error_message
  ):
    multitask_items = [
        tfdf.keras.MultiTaskItem(
            label='task_1', task=tfdf.keras.Task.CLASSIFICATION, primary=True
        ),
        tfdf.keras.MultiTaskItem(
            label='task_2', task=tfdf.keras.Task.REGRESSION, primary=False
        ),
    ]

    train_x = {'feature_1': tf.constant([1.0, 2.0, 3.0])}

    with self.assertRaisesRegex(expected_error, expected_error_message):
      _, _, _ = training_preprocessing.multitask_label_presence_processing(
          multitask_items
      )(train_x, train_y, train_weights)

  def test_training_multi_task_model_with_preprocessing(self):
    num_features = 5

    def make_dataset(num_examples):
      features = np.random.uniform(size=(num_examples, num_features))
      hidden = 0.05 * np.random.uniform(size=num_examples)

      # Binary classification.
      label_1 = (hidden >= features[:, 1]).astype(int) + (
          hidden >= features[:, 2]
      ).astype(int)

      # Regression
      label_2 = hidden + features[:, 2] + 2 * features[:, 3]

      # Multi-class classification.
      label_3 = (
          (np.random.uniform(size=num_examples) + features[:, 3]) * 4 / 2
      ).astype(int)

      labels = {
          'task_1': label_1,
          'task_2': label_2,
          'task_3': label_3,
      }

      sample_weight_1 = np.random.binomial(n=1, p=0.8, size=num_examples)
      sample_weight_2 = np.random.binomial(n=1, p=0.8, size=num_examples)
      sample_weight_3 = np.random.binomial(n=1, p=0.8, size=num_examples)
      sample_weights = {
          'task_1': sample_weight_1,
          'task_2': sample_weight_2,
          'task_3': sample_weight_3,
      }
      return tf.data.Dataset.from_tensor_slices(
          (features, labels, sample_weights)
      ).batch(3)

    train_dataset = make_dataset(1000)
    test_dataset = make_dataset(10)

    multitask_items = [
        tfdf.keras.MultiTaskItem(
            label='task_1',
            task=tfdf.keras.Task.CLASSIFICATION,
            primary=False,
        ),
        tfdf.keras.MultiTaskItem(
            label='task_2', task=tfdf.keras.Task.REGRESSION, primary=False
        ),
        tfdf.keras.MultiTaskItem(
            label='task_3', task=tfdf.keras.Task.CLASSIFICATION
        ),
    ]
    multitask_preprocessing = (
        training_preprocessing.multitask_label_presence_processing(
            multitask_items
        )
    )

    model = tfdf.keras.GradientBoostedTreesModel(
        multitask=multitask_items,
        advanced_arguments=tfdf.keras.AdvancedArguments(
            populate_history_with_yggdrasil_logs=True,
            output_secondary_class_predictions=True,
        ),
        training_preprocessing=multitask_preprocessing,
        verbose=2,
    )
    model.fit(train_dataset)

    logging.info('model:')
    model.summary()

    # Check model
    inspector_0 = model.make_inspector(0)
    inspector_1 = model.make_inspector(1)
    inspector_2 = model.make_inspector(2)

    self.assertEqual(inspector_0.model_type(), 'GRADIENT_BOOSTED_TREES')
    self.assertEqual(inspector_0.label().name, 'task_1')

    self.assertEqual(inspector_1.model_type(), 'GRADIENT_BOOSTED_TREES')
    self.assertEqual(inspector_1.label().name, 'task_2')

    self.assertEqual(inspector_2.model_type(), 'GRADIENT_BOOSTED_TREES')
    self.assertEqual(inspector_2.label().name, 'task_3')

    # Check predictions
    predictions = model.predict(test_dataset)
    logging.info('predictions: %s', predictions)
    self.assertSetEqual(
        set(predictions.keys()), set(['task_1', 'task_2', 'task_3'])
    )
    self.assertTupleEqual(predictions['task_1'].shape, (10, 1))
    self.assertTupleEqual(predictions['task_2'].shape, (10, 1))
    self.assertTupleEqual(predictions['task_3'].shape, (10, 4))

    # Export / import model
    saved_model_path = os.path.join(self.get_temp_dir(), 'saved_model')
    logging.info('Saving model to %s', saved_model_path)
    model.save(saved_model_path)

    logging.info('Loading model from %s', saved_model_path)
    loaded_model = tf.keras.models.load_model(saved_model_path)
    loaded_model.summary()

    # Check exported / imported model predictions
    loaded_predictions = loaded_model.predict(test_dataset)
    logging.info('loaded predictions: %s', loaded_predictions)

    self.assertEqual(predictions, loaded_predictions)


if __name__ == '__main__':
  tf.test.main()
