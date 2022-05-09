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
import numpy as np
import tensorflow as tf

from tensorflow_decision_forests import keras
from tensorflow_decision_forests.component import py_tree
from tensorflow_decision_forests.component.inspector import inspector as insp
from yggdrasil_decision_forests.metric import metric_pb2
from yggdrasil_decision_forests.model.gradient_boosted_trees import gradient_boosted_trees_pb2

ColumnType = insp.ColumnType
SimpleColumnSpec = insp.SimpleColumnSpec
CATEGORICAL = insp.ColumnType.CATEGORICAL
NUMERICAL = insp.ColumnType.NUMERICAL


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


class InspectorTest(parameterized.TestCase, tf.test.TestCase):

  def test_classification_random_forest(self):
    model_path = os.path.join(test_model_directory(), "adult_binary_class_rf")
    # dataset_path = os.path.join(test_dataset_directory(), "adult_test.csv")
    inspector = insp.make_inspector(model_path)

    self.assertEqual(inspector.model_type(), "RANDOM_FOREST")
    self.assertEqual(inspector.task, insp.Task.CLASSIFICATION)
    self.assertEqual(inspector.num_trees(), 100)
    self.assertEqual(inspector.label(),
                     SimpleColumnSpec("income", CATEGORICAL, 14))
    self.assertEqual(
        inspector.objective(),
        py_tree.objective.ClassificationObjective(
            label="income", classes=["<=50K", ">50K"]))

    self.assertEqual(inspector.features(), [
        SimpleColumnSpec("age", NUMERICAL, 0),
        SimpleColumnSpec("workclass", CATEGORICAL, 1),
        SimpleColumnSpec("fnlwgt", NUMERICAL, 2),
        SimpleColumnSpec("education", CATEGORICAL, 3),
        SimpleColumnSpec("education_num", CATEGORICAL, 4),
        SimpleColumnSpec("marital_status", CATEGORICAL, 5),
        SimpleColumnSpec("occupation", CATEGORICAL, 6),
        SimpleColumnSpec("relationship", CATEGORICAL, 7),
        SimpleColumnSpec("race", CATEGORICAL, 8),
        SimpleColumnSpec("sex", CATEGORICAL, 9),
        SimpleColumnSpec("capital_gain", NUMERICAL, 10),
        SimpleColumnSpec("capital_loss", NUMERICAL, 11),
        SimpleColumnSpec("hours_per_week", NUMERICAL, 12),
        SimpleColumnSpec("native_country", CATEGORICAL, 13),
    ])

    self.assertEqual(inspector.evaluation().num_examples, 22792)

    self.assertAlmostEqual(
        inspector.evaluation().accuracy, 0.86512, delta=0.0001)

    self.assertLen(inspector.training_logs(), 2)
    self.assertAlmostEqual(
        inspector.training_logs()[-1].evaluation.accuracy,
        0.86512,
        delta=0.0001)

    self.assertEqual(inspector.training_logs()[-1].num_trees,
                     inspector.num_trees())

    self.assertEqual(inspector.winner_take_all_inference(), False)

    variable_importances = inspector.variable_importances()
    self.assertEqual(
        variable_importances, {
            "NUM_AS_ROOT": [
                (SimpleColumnSpec("relationship", CATEGORICAL, 7), 33.0),
                (SimpleColumnSpec("marital_status", CATEGORICAL, 5), 28.0),
                (SimpleColumnSpec("capital_gain", NUMERICAL, 10), 15.0),
                (SimpleColumnSpec("education_num", CATEGORICAL, 4), 11.0),
                (SimpleColumnSpec("age", NUMERICAL, 0), 6.0),
                (SimpleColumnSpec("education", CATEGORICAL, 3), 4.0),
                (SimpleColumnSpec("occupation", CATEGORICAL, 6), 3.0)
            ]
        })

    num_nodes = 0
    for _ in inspector.iterate_on_nodes():
      num_nodes += 1
    self.assertEqual(num_nodes, 125578)

    tree = inspector.extract_tree(tree_idx=1)  # Second tree
    logging.info("Tree:\n%s", tree)

    # Checked with :show_model --full_definition
    self.assertEqual(tree.root.condition.feature.name, "capital_gain")

    all_trees = inspector.extract_all_trees()
    self.assertLen(all_trees, inspector.num_trees())
    self.assertEqual(all_trees[1].root.condition.feature.name, "capital_gain")

    tensorboard_logs = os.path.join(tmp_path(), "tensorboard_logs")
    inspector.export_to_tensorboard(tensorboard_logs)

    logging.info("tensorboard_logs: %s", tensorboard_logs)

    # Note: The tree has been partially checked manually against ":show_model".
    # :show_model --full_definition --model=...
    self.assertEqual(
        tree.pretty(),
        """(capital_gain >= 7073.5; miss=False, score=0.06218588352203369)
    ├─(pos)─ (education in ['Some-college', 'Bachelors', 'Masters', 'Assoc-voc', '11th', 'Assoc-acdm', '7th-8th', 'Prof-school', '12th', 'Doctorate', '5th-6th']; miss=False, score=0.019492337480187416)
    │        ├─(pos)─ (age >= 62.5; miss=False, score=0.005498059559613466)
    │        │        ├─(pos)─ (hours_per_week >= 49.0; miss=False, score=0.029734181240200996)
    │        │        │    ...
    │        │        └─(neg)─ ProbabilityValue([0.0, 1.0],n=785.0) (idx=596)
    │        └─(neg)─ (age >= 34.5; miss=True, score=0.049639273434877396)
    │                 ├─(pos)─ (workclass in ['Private', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov']; miss=True, score=0.06564109027385712)
    │                 │    ...
    │                 └─(neg)─ (hours_per_week >= 37.5; miss=True, score=0.6460905075073242)
    │                      ...
    └─(neg)─ (marital_status in ['Married-civ-spouse', 'Married-AF-spouse']; miss=True, score=0.10465919226408005)
             ├─(pos)─ (education_num in ['13', '14', '15', '16']; miss=False, score=0.057999979704618454)
             │        ├─(pos)─ (capital_loss >= 1782.5; miss=False, score=0.03205965831875801)
             │        │    ...
             │        └─(neg)─ (native_country in ['United-States', 'Philippines', 'Germany', 'Canada', 'Cuba', 'England', 'Jamaica', 'Dominican-Republic', 'South', 'Italy', 'Taiwan', 'Poland', 'Iran', 'Haiti', 'Peru', 'France', 'Ecuador', 'Thailand', 'Cambodia', 'Ireland', 'Yugoslavia', 'Trinadad&Tobago', 'Hungary', 'Hong']; miss=True, score=0.01416037604212761)
             │             ...
             └─(neg)─ (age >= 27.5; miss=True, score=0.01477713044732809)
                      ├─(pos)─ (hours_per_week >= 40.5; miss=False, score=0.01574750244617462)
                      │    ...
                      └─(neg)─ (hours_per_week >= 58.5; miss=False, score=0.004579735919833183)
                           ...
Label classes: ['<=50K', '>50K']
""")

  def test_regression_random_forest(self):
    model_path = os.path.join(test_model_directory(), "abalone_regression_rf")
    # dataset_path = os.path.join(test_dataset_directory(), "abalone.csv")
    inspector = insp.make_inspector(model_path)

    self.assertEqual(inspector.model_type(), "RANDOM_FOREST")
    self.assertEqual(inspector.task, insp.Task.REGRESSION)
    self.assertEqual(inspector.num_trees(), 100)
    self.assertEqual(inspector.label(), SimpleColumnSpec("Rings", NUMERICAL, 8))
    self.assertEqual(inspector.evaluation().num_examples, 2940)
    self.assertAlmostEqual(inspector.evaluation().rmse, 2.13434, delta=0.0001)
    self.assertEqual(inspector.objective(),
                     py_tree.objective.RegressionObjective(label="Rings"))

    num_nodes = 0
    for _ in inspector.iterate_on_nodes():
      num_nodes += 1
    self.assertEqual(num_nodes, 88494)

    tree = inspector.extract_tree(tree_idx=10)
    logging.info("Tree:\n%s", tree)

  def test_classification_gradient_boosted_tree(self):

    n = 1000
    features = np.random.normal(size=[n, 3])
    labels = features[:, 0] + features[:, 1] + np.random.normal(size=n) >= 0

    # Early stopping will trigger before all the trees are trained.
    model = keras.GradientBoostedTreesModel(num_trees=10000)
    model.fit(x=features, y=labels)

    inspector = model.make_inspector()

    # Because of early stopping, the training logs contains the evaluation of
    # more trees than what is in the final model.
    self.assertGreater(inspector.training_logs()[-1].num_trees,
                       inspector.num_trees())

    # It is very unlikely that the model contains less than 10 trees.
    self.assertGreater(inspector.num_trees(), 10)

    self.assertAlmostEqual(inspector.bias, -0.023836, delta=0.0001)
    self.assertEqual(inspector.num_trees_per_iter, 1)

    matching_log = [
        log for log in inspector.training_logs()
        if log.num_trees == inspector.num_trees()
    ]
    self.assertLen(matching_log, 1)
    self.assertEqual(matching_log[0].evaluation, inspector.evaluation())

  @parameterized.parameters(
      {
          "model": "adult_binary_class_gbdt",
          "dataset": "adult_test.csv",
          "model_name": "GRADIENT_BOOSTED_TREES",
          "task": insp.Task.CLASSIFICATION
      },
      {
          "model": "adult_binary_class_oblique_rf",
          "dataset": "adult_test.csv",
          "model_name": "RANDOM_FOREST",
          "task": insp.Task.CLASSIFICATION
      },
      {
          "model": "adult_binary_class_rf_discret_numerical",
          "dataset": "adult_test.csv",
          "model_name": "RANDOM_FOREST",
          "task": insp.Task.CLASSIFICATION
      },
      {
          "model": "sst_binary_class_gbdt",
          "dataset": "sst_binary_test.csv",
          "model_name": "GRADIENT_BOOSTED_TREES",
          "task": insp.Task.CLASSIFICATION
      },
      {
          "model": "synthetic_ranking_gbdt",
          "dataset": "synthetic_ranking_test.csv",
          "model_name": "GRADIENT_BOOSTED_TREES",
          "task": insp.Task.RANKING
      },
  )
  def test_generic(self, model, dataset, model_name, task):
    model_path = os.path.join(test_model_directory(), model)
    inspector = insp.make_inspector(model_path)

    self.assertEqual(inspector.model_type(), model_name)
    self.assertEqual(inspector.task, task)

    if model != "sst_binary_class_gbdt":
      # Computing the variable importance of the SST model takes a lot of time.
      logging.info("Variable importances:\n%s",
                   inspector.variable_importances())

    logging.info("Evaluation:\n%s", inspector.evaluation())
    logging.info("Training logs:\n%s", inspector.training_logs())

    num_nodes = 0
    for _ in inspector.iterate_on_nodes():
      num_nodes += 1
      if num_nodes > 1000:
        break

    tree = inspector.extract_tree(tree_idx=2)
    logging.info("Tree:\n%s", tree)

    tensorboard_logs = os.path.join(tmp_path(), "tensorboard_logs")
    inspector.export_to_tensorboard(tensorboard_logs)

  def test_proto_evaluation_to_evaluation(self):
    evaluation = metric_pb2.EvaluationResults()
    evaluation.count_predictions_no_weight = 10
    evaluation.count_predictions = 10
    confusion = evaluation.classification.confusion
    confusion.nrow = 3
    confusion.ncol = 3
    confusion.counts[:] = [2, 1, 1, 1, 3, 1, 1, 1, 4]
    confusion.sum = 15
    roc_0 = evaluation.classification.rocs.add()
    roc_0.auc = 0.6
    roc_1 = evaluation.classification.rocs.add()
    roc_1.auc = 0.8
    roc_2 = evaluation.classification.rocs.add()
    roc_2.auc = 0.9
    self.assertEqual(
        insp._proto_evaluation_to_evaluation(evaluation),
        insp.Evaluation(
            num_examples=10, accuracy=(2 + 3 + 4) / 15.0, aucs=[0.6, 0.8, 0.9]))

    evaluation = metric_pb2.EvaluationResults()
    evaluation.count_predictions_no_weight = 10
    evaluation.count_predictions = 10
    evaluation.loss_value = 5
    evaluation.regression.sum_square_error = 10
    self.assertEqual(
        insp._proto_evaluation_to_evaluation(evaluation),
        insp.Evaluation(num_examples=10, loss=5.0, rmse=1.0))

    evaluation = metric_pb2.EvaluationResults()
    evaluation.count_predictions_no_weight = 10
    evaluation.count_predictions = 10
    evaluation.ranking.ndcg.value = 10
    self.assertEqual(
        insp._proto_evaluation_to_evaluation(evaluation),
        insp.Evaluation(num_examples=10, ndcg=10.0))

  def test_gbt_log_entry_to_evaluation(self):
    logs = gradient_boosted_trees_pb2.TrainingLogs()
    logs.secondary_metric_names[:] = ["accuracy", "NDCG@5"]
    logs.entries.add()  # One empty entry.
    entry = logs.entries.add()
    entry.validation_loss = 0.1
    entry.validation_secondary_metrics[:] = [0.2, 0.3]

    self.assertAlmostEqual(
        insp._gbt_log_entry_to_evaluation(logs, 1).loss, 0.1, delta=0.0001)
    self.assertAlmostEqual(
        insp._gbt_log_entry_to_evaluation(logs, 1).accuracy, 0.2, delta=0.0001)
    self.assertAlmostEqual(
        insp._gbt_log_entry_to_evaluation(logs, 1).ndcg, 0.3, delta=0.0001)


if __name__ == "__main__":
  tf.test.main()
