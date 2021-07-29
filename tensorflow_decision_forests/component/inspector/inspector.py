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

"""Model inspector.

Utility to access the structure and meta-data (e.g. variable importance,
training logs) of a model.

Usage:

```
model = keras.RandomForest().
model.fit(...)
inspector = model.make_inspector()

# Or
inspector = make_inspector(<model directory>)

print(inspector.name())
print(inspector.num_trees())
# Note: "inspector"'s accessors depends on the model type (inspector.name()).
```

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import math
import os
import typing
from typing import List, Any, Optional, Generator, Callable, Dict, Tuple

import six
import tensorflow as tf

from tensorflow_decision_forests.component import py_tree
from tensorflow_decision_forests.component.inspector import blob_sequence
from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.metric import metric_pb2
from yggdrasil_decision_forests.model import abstract_model_pb2
from yggdrasil_decision_forests.model.decision_tree import decision_tree_pb2
from yggdrasil_decision_forests.model.gradient_boosted_trees import gradient_boosted_trees_pb2
from yggdrasil_decision_forests.model.random_forest import random_forest_pb2

# Filenames used in Yggdrasil models.
FILENAME_HEADER = "header.pb"
FILENAME_DATASPEC = "data_spec.pb"
FILENAME_NODES_SHARD = "nodes"
FILENAME_DONE = "done"

Task = abstract_model_pb2.Task
ColumnType = data_spec_pb2.ColumnType
SimpleColumnSpec = py_tree.dataspec.SimpleColumnSpec


def make_inspector(directory: str) -> "AbstractInspector":
  """Creates an inspector for a model saved in a directory."""

  # Determine the format of the model.
  header = abstract_model_pb2.AbstractModel()
  with tf.io.gfile.GFile(os.path.join(directory, FILENAME_HEADER), "rb") as f:
    header.ParseFromString(f.read())
  if header.name not in MODEL_INSPECTORS:
    raise ValueError(
        f"The model type {header.name} is not supported by the inspector. The "
        "supported types are: {MODEL_INSPECTORS.keys()}")

  # Create the inspector.
  return MODEL_INSPECTORS[header.name](directory)


class IterNodeResult(typing.NamedTuple):
  """Value returned by node iterator methods. See "iterate_on_nodes"."""

  # Note without children.
  node: py_tree.node.AbstractNode
  # Depth of the node. Depth=0 for the root node.
  depth: int
  # Index of the tree.
  tree_idx: int


class Evaluation(typing.NamedTuple):
  """Evaluation of a model."""

  num_examples: Optional[int] = None
  accuracy: Optional[float] = None
  loss: Optional[float] = None
  rmse: Optional[float] = None
  ndcg: Optional[float] = None
  aucs: Optional[List[float]] = None


class TrainLog(typing.NamedTuple):
  """One entry in the training logs of a model."""

  num_trees: int
  evaluation: Optional[Evaluation] = None


@six.add_metaclass(abc.ABCMeta)
class AbstractInspector(object):
  """Abstract inspector for all Yggdrasil models."""

  def __init__(self, directory: str):
    self._directory = directory

    self._header = _read_binary_proto(abstract_model_pb2.AbstractModel,
                                      os.path.join(directory, FILENAME_HEADER))

    self._dataspec = _read_binary_proto(
        data_spec_pb2.DataSpecification,
        os.path.join(directory, FILENAME_DATASPEC))

  @abc.abstractmethod
  def model_type(self) -> str:
    """Unique key describing the type of the model.

    Note that different learners can output similar model types, and a given
    learner can output different model types.
    """

    raise NotImplementedError("Must be implemented in subclasses.")

  @property
  def task(self) -> Task:
    """Task solved by the model."""

    return self._header.task

  def objective(self) -> py_tree.objective.AbstractObjective:
    """Objective solved by the model i.e. Task + extra information."""

    label = self.label()

    if self.task == Task.CLASSIFICATION:
      label_column = self._dataspec.columns[self._header.label_col_idx]
      if label_column.type != ColumnType.CATEGORICAL:
        raise ValueError("Categorical type expected for classification label."
                         f" Got {label_column.type} instead.")

      if label_column.categorical.is_already_integerized:
        return py_tree.objective.ClassificationObjective(
            label=label.name,
            num_classes=label_column.categorical.number_of_unique_values + 1)
      else:
        # The first element is the "out-of-vocabulary" that is not used in
        # labels.
        label_classes = py_tree.dataspec.categorical_column_dictionary_to_list(
            label_column)[1:]
        return py_tree.objective.ClassificationObjective(
            label=label.name, classes=label_classes)

    elif self.task == Task.REGRESSION:
      return py_tree.objective.RegressionObjective(label=label.name)

    elif self.task == Task.RANKING:
      group_column = self._dataspec.columns[self._header.ranking_group_col_idx]
      return py_tree.objective.RankingObjective(
          label=label.name, group=group_column.name)

    else:
      raise NotImplementedError()

  def features(self) -> List[py_tree.dataspec.SimpleColumnSpec]:
    """Input features of the model."""

    return [
        self._make_simple_column_spec(col_idx)
        for col_idx in self._header.input_features
    ]

  def label(self) -> py_tree.dataspec.SimpleColumnSpec:
    """Label predicted by the model."""

    return self._make_simple_column_spec(self._header.label_col_idx)

  def label_classes(self) -> Optional[List[str]]:
    """Possible classes of the label.

    If the task is not a classification, or if the labels are dense integers,
    returns None.

    Returns:
      The list of label values, or None.
    """

    if self.task != Task.CLASSIFICATION:
      return None

    label_column = self._dataspec.columns[self._header.label_col_idx]
    if label_column.type != ColumnType.CATEGORICAL:
      raise ValueError("Categorical type expected for classification label."
                       f" Got {label_column.type} instead.")

    if label_column.categorical.is_already_integerized:
      return None

    # The first element is the "out-of-vocabulary" that is not used in labels.
    return py_tree.dataspec.categorical_column_dictionary_to_list(
        label_column)[1:]

  def variable_importances(
      self) -> Dict[str, List[Tuple[py_tree.dataspec.SimpleColumnSpec, float]]]:
    """Various variable importances.

    Values are sorted by decreasing value/importance.

    The importance of a variable indicates how much a variable contributes to
    the model predictions or to the model quality.

    The available variable importances depends on the model type and possibly
    its hyper-parameters.

    Returns:
      Variable importances.
    """

    vis = {}
    # Collect the variable importances stored in the model.
    for name, importance in self._header.precomputed_variable_importances.items(
    ):
      vis[name] = [(self._make_simple_column_spec(src.attribute_idx),
                    src.importance) for src in importance.variable_importances]
    return vis

  def evaluation(self) -> Optional[Evaluation]:
    """Model self evaluation.

    The model self evaluation is a cheap alternative to the use of a separate
    validation dataset or cross-validation. The exact implementation depends on
    the model e.g. Out-of-bag evaluation, internal train-validation.

    During training, some models (e.g. Gradient Boosted Tree) used this
    evaluation for early stopping (if early stopping is enabled).

    While this evaluation is  computed during training, it can be used as a low
    quality model evaluation.

    Returns:
      The evaluation, or None is not evaluation is available.
    """

    return None

  def training_logs(self) -> Optional[List[TrainLog]]:
    """Evaluation metrics and statistics about the model during training.

    The training logs show the quality of the model (e.g. accuracy evaluated on
    the out-of-bag or validation dataset) according to the number of trees in
    the model. Logs are useful to characterize the balance between model size
    and model quality.
    """

    return None

  def export_to_tensorboard(self, path: str) -> None:
    """Export the training logs (and possibly other metadata) to TensorBoard.

    Usage examples in Colab:

    ```python
    model.make_inspector().export_to_tensorboard("/tmp/tensorboard_logs")
    %load_ext tensorboard
    %tensorboard --logdir "/tmp/tensorboard_logs"
    ```

    Note that you can compare multiple models runs using sub-directories. For
    examples:

    ```python
    model_1.make_inspector().export_to_tensorboard("/tmp/tb_logs/model_1")
    model_2.make_inspector().export_to_tensorboard("/tmp/tb_logs/model_2")

    %load_ext tensorboard
    %tensorboard --logdir "/tmp/tb_logs"
    ```

    Args:
      path: Output directory for the logs.
    """

    writer = tf.summary.create_file_writer(path)
    with writer.as_default():

      tf.summary.text("model_type", self.model_type(), step=0)

      evaluation = self.evaluation()
      if evaluation:
        for key, value in evaluation._asdict().items():
          if value is None:
            continue
          tf.summary.scalar(
              "final/" + key,
              value,
              step=0,
              description=f"{key}'s evaluation of the model after training. "
              "Note that because of rollback early stopping (or other "
              "mechanisms), the final evaluation is not necessary the last one "
              "in the training logs.")

      logs = self.training_logs() or []
      for log in logs:
        for key, value in log.evaluation._asdict().items():
          if value is None:
            continue
          tf.summary.scalar(key, value, step=log.num_trees)
      writer.flush()

  @property
  def dataspec(self) -> data_spec_pb2.DataSpecification:
    """Gets the dataspec."""

    return self._dataspec

  def _make_simple_column_spec(
      self, col_idx: int) -> py_tree.dataspec.SimpleColumnSpec:
    """Creates a SimpleColumnSpec using the model's dataspec."""

    return py_tree.dataspec.make_simple_column_spec(self._dataspec, col_idx)


@six.add_metaclass(abc.ABCMeta)
class _AbstractDecisionForestInspector(AbstractInspector):
  """Abstract inspector for decision forest models."""

  def num_trees(self) -> int:
    """Gets the number of trees contained in the model."""

    return self.specialized_header().num_trees

  @abc.abstractmethod
  def specialized_header(self):
    """Gets the specialized header.

    The specialized header is a header proto specific to a model type.
    """

    pass

  def variable_importances(
      self) -> Dict[str, List[Tuple[py_tree.dataspec.SimpleColumnSpec, float]]]:
    """Various definitions of variable importances.

    If addition to the model generic variable importances and the variable
    importances contained in the model, adds:

      - The number of times a feature is used in the root node.

    Returns:
      Variable importances.
    """

    core_vs = super(_AbstractDecisionForestInspector,
                    self).variable_importances()

    if "NUM_AS_ROOT" not in core_vs:
      num_as_root = collections.defaultdict(lambda: 0.0)

      for node_iter in self.iterate_on_nodes():
        if node_iter.depth == 0 and isinstance(node_iter.node,
                                               py_tree.node.NonLeafNode):
          # This is a root node with a condition.
          for attribute in node_iter.node.condition.features():
            num_as_root[attribute] += 1

      core_vs["NUM_AS_ROOT"] = _variable_importance_dict_to_list(num_as_root)

    return core_vs

  def iterate_on_nodes(self) -> Generator[IterNodeResult, None, None]:
    """Creates a generator over all the nodes.

    The nodes of the model are visited with a Depth First Pre-order traversals,
    one tree after another.

    Yields:
      Depth First Pre-order traversals over the nodes.
    """

    num_shards = self.specialized_header().num_node_shards
    tree_idx = 0
    # Sequence of positive/negative branches to the current node.
    branch: List[bool] = []

    for shard_idx in range(num_shards):
      shard_path = os.path.join(
          self._directory, "{}-{:05d}-of-{:05d}".format(FILENAME_NODES_SHARD,
                                                        shard_idx, num_shards))

      for serialized_node in _create_node_reader(
          self.specialized_header().node_format, shard_path):
        core_node = decision_tree_pb2.Node.FromString(serialized_node)
        node = py_tree.node.core_node_to_node(core_node, self._dataspec)
        yield IterNodeResult(node=node, depth=len(branch), tree_idx=tree_idx)

        if isinstance(node, py_tree.node.NonLeafNode):
          branch.append(False)

        else:
          # Unroll to the last negative branch.
          while branch and branch[-1]:
            branch.pop()

          if not branch:
            # New tree
            tree_idx += 1
          else:
            branch[-1] = True

  def extract_tree(self, tree_idx: int) -> py_tree.tree.Tree:
    """Extracts a decision tree.

    This operation materializes the decision tree using the python
    representation. When possible, for efficiency reasons, use
    "iterate_on_nodes" or implement your algorithm in C++.

    Args:
      tree_idx: Index of the tree to extract. Should be in [0, num_trees()).

    Returns:
      The extracted tree.
    """

    node_generator = self.iterate_on_nodes()
    for _ in range(tree_idx):
      _extract_branch(node_generator)

    return py_tree.tree.Tree(
        root=_extract_branch(node_generator),
        label_classes=self.label_classes())

  def extract_all_trees(self) -> List[py_tree.tree.Tree]:
    """Extracts all the decision trees of the model.

    This method is more efficient than calling "extract_tree" repeatedly. See
    "extract_tree" for more details.

    Returns:
      The list of extracted trees.
    """

    node_generator = self.iterate_on_nodes()

    trees = []
    for _ in range(self.num_trees()):
      trees.append(
          py_tree.tree.Tree(
              root=_extract_branch(node_generator),
              label_classes=self.label_classes()))

    return trees


class _RandomForestInspector(_AbstractDecisionForestInspector):
  """Inspector for the RANDOM_FOREST model."""

  MODEL_NAME = "RANDOM_FOREST"

  def __init__(self, directory: str):
    super(_RandomForestInspector, self).__init__(directory)

    self._specialized_header = _read_binary_proto(
        random_forest_pb2.Header,
        os.path.join(directory, "random_forest_header.pb"))

  def model_type(self) -> str:
    return _RandomForestInspector.MODEL_NAME

  def specialized_header(self):
    return self._specialized_header

  def winner_take_all_inference(self):
    """Does the model use a winner take all voting strategy for classification.

    If true, each tree votes for a single class.
    If false, each tree outputs a probability distribution over all the classes.

    Returns:
      Is the winner-take-all voting strategy used?
    """

    return self._specialized_header.winner_take_all_inference

  def evaluation(self) -> Optional[Evaluation]:
    if not self._specialized_header.out_of_bag_evaluations:
      return None

    return _proto_evaluation_to_evaluation(
        self._specialized_header.out_of_bag_evaluations[-1].evaluation)

  def training_logs(self) -> Optional[List[TrainLog]]:
    if not self._specialized_header.out_of_bag_evaluations:
      return None

    return [
        TrainLog(
            num_trees=log.number_of_trees,
            evaluation=_proto_evaluation_to_evaluation(log.evaluation))
        for log in self._specialized_header.out_of_bag_evaluations
    ]


class _GradientBoostedTreeInspector(_AbstractDecisionForestInspector):
  """Inspector for the GRADIENT_BOOSTED_TREE model."""

  MODEL_NAME = "GRADIENT_BOOSTED_TREES"

  def __init__(self, directory: str):
    super(_GradientBoostedTreeInspector, self).__init__(directory)

    self._specialized_header = _read_binary_proto(
        gradient_boosted_trees_pb2.Header,
        os.path.join(directory, "gradient_boosted_trees_header.pb"))

  def model_type(self) -> str:
    return _GradientBoostedTreeInspector.MODEL_NAME

  def specialized_header(self):
    return self._specialized_header

  def evaluation(self) -> Optional[Evaluation]:
    if not self._specialized_header.HasField("training_logs"):
      return Evaluation(loss=self._specialized_header.validation_loss)

    # Find the training log that correspond to the final model.
    logs = self._specialized_header.training_logs
    final_log_idxs = [
        entry_idx for entry_idx, entry in enumerate(logs.entries)
        if logs.number_of_trees_in_final_model == entry.number_of_trees
    ]
    if not final_log_idxs:
      return None

    return _gbt_log_entry_to_evaluation(logs, final_log_idxs[0])

  def training_logs(self) -> Optional[List[TrainLog]]:
    if not self._specialized_header.HasField("training_logs"):
      return None

    logs = self._specialized_header.training_logs
    return [
        TrainLog(
            num_trees=entry.number_of_trees,
            evaluation=_gbt_log_entry_to_evaluation(logs, entry_idx))
        for entry_idx, entry in enumerate(logs.entries)
    ]


def _extract_branch(
    node_generator: Generator[IterNodeResult, None, None]
) -> Optional[py_tree.node.AbstractNode]:
  """Extracts a branch (i.e. node and children) from a sequence of node."""

  node = next(node_generator).node
  if isinstance(node, py_tree.node.NonLeafNode):
    node.neg_child = _extract_branch(node_generator)
    node.pos_child = _extract_branch(node_generator)
  return node


def _create_node_reader(container_format: str, path: str) -> Any:
  """Creates a sequential node reader."""

  if container_format == "BLOB_SEQUENCE":
    return blob_sequence.Reader(path)

  raise ValueError(f"Unknown node format {container_format}.")


def _read_binary_proto(proto_type: Callable[[], Any], path: str) -> Any:
  """Reads a binary serialized proto from disk."""

  proto = proto_type()
  with tf.io.gfile.GFile(path, "rb") as f:
    proto.ParseFromString(f.read())
  return proto


def _variable_importance_dict_to_list(
    src: Dict[py_tree.dataspec.SimpleColumnSpec, float]
) -> List[Tuple[py_tree.dataspec.SimpleColumnSpec, float]]:
  """Converts a variable importance from a dictionary into a list."""

  dst = list(src.items())
  dst.sort(key=lambda x: x[1], reverse=True)
  return dst


def _proto_evaluation_to_evaluation(
    src: metric_pb2.EvaluationResults) -> Evaluation:
  """Converts an Yggdrasil evaluation proto to an evaluation object."""

  dst = Evaluation(num_examples=src.count_predictions_no_weight)

  if dst.num_examples == 0:
    return dst

  if src.HasField("loss_value"):
    dst = dst._replace(loss=src.loss_value)

  if src.HasField("classification"):
    cls = src.classification
    if cls.HasField("confusion"):
      sum_diagonal = 0
      for i in range(cls.confusion.nrow):
        sum_diagonal += cls.confusion.counts[i + i * cls.confusion.nrow]
      dst = dst._replace(accuracy=sum_diagonal / cls.confusion.sum)

    if cls.rocs:
      dst = dst._replace(aucs=[roc.auc for roc in cls.rocs])

    if dst.loss is None and cls.HasField("sum_log_loss"):
      dst = dst._replace(loss=cls.sum_log_loss / src.count_predictions)

  if src.HasField("regression"):
    reg = src.regression
    if reg.HasField("sum_square_error"):
      dst = dst._replace(
          rmse=math.sqrt(reg.sum_square_error / src.count_predictions))

  if src.HasField("ranking"):
    rank = src.ranking
    if rank.HasField("ndcg"):
      dst = dst._replace(ndcg=rank.ndcg.value)

  return dst


def _gbt_log_entry_to_evaluation(logs: gradient_boosted_trees_pb2.TrainingLogs,
                                 entry_idx: int) -> Evaluation:
  """Converts a GBT log entry into an evaluation."""

  final_log = logs.entries[entry_idx]
  evaluation = Evaluation(loss=final_log.validation_loss)

  for metric_idx, metric_key in enumerate(logs.secondary_metric_names):
    value = final_log.validation_secondary_metrics[metric_idx]
    if metric_key == "accuracy":
      evaluation = evaluation._replace(accuracy=value)
    elif metric_key == "NDCG@5":
      evaluation = evaluation._replace(ndcg=value)
    elif metric_key == "rmse":
      evaluation = evaluation._replace(rmse=value)

  return evaluation


MODEL_INSPECTORS = {
    cls.MODEL_NAME: cls
    for cls in [_RandomForestInspector, _GradientBoostedTreeInspector]
}
