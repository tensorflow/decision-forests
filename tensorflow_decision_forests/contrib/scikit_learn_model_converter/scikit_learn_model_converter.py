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

"""Utilities for converting Scikit-Learn models into Tensorflow models."""

import contextlib
import enum
import functools
import os
import tempfile
from typing import Any, Dict, List, Optional, TypeVar, Union

from sklearn import base
from sklearn import dummy
from sklearn import ensemble
from sklearn import tree
import tensorflow as tf
import tensorflow_decision_forests as tfdf


class TaskType(enum.Enum):
  """The type of task that a scikit-learn model performs."""
  UNKNOWN = 1
  SCALAR_REGRESSION = 2
  SINGLE_LABEL_CLASSIFICATION = 3


ScikitLearnModel = TypeVar("ScikitLearnModel", bound=base.BaseEstimator)
ScikitLearnTree = TypeVar("ScikitLearnTree", bound=tree.BaseDecisionTree)


def convert(
    sklearn_model: ScikitLearnModel,
    intermediate_write_path: Optional[os.PathLike] = None,
) -> tf.keras.Model:
  """Converts a tree-based scikit-learn model to a tensorflow model.

  Currently supported models are:
  *   sklearn.tree.DecisionTreeClassifier
  *   sklearn.tree.DecisionTreeRegressor
  *   sklearn.tree.ExtraTreeClassifier
  *   sklearn.tree.ExtraTreeRegressor
  *   sklearn.ensemble.RandomForestClassifier
  *   sklearn.ensemble.RandomForestRegressor
  *   sklearn.ensemble.ExtraTreesClassifier
  *   sklearn.ensemble.ExtraTreesRegressor
  *   sklearn.ensemble.GradientBoostingRegressor

  Additionally, only single-label classification and scalar regression are
  supported (e.g. multivariate regression models will not convert).

  Args:
    sklearn_model: the scikit-learn tree based model to be converted.
    intermediate_write_path: path to a directory. As part of the conversion
      process, a TFDF model is written to disk. If intermediate_write_path is
      specified, the TFDF model is written to this directory. Otherwise, a
      temporary directory is created that is immediately removed after this
      function executes. Note that in order to save the converted model and
      load it again later, this argument must be provided.

  Returns:
    a keras Model that emulates the provided scikit-learn model.
  """
  if not intermediate_write_path:
    # No intermediate directory was provided, so this creates one using the
    # TemporaryDirectory context mananger, which handles teardown.
    intermediate_write_directory = tempfile.TemporaryDirectory()
    path = intermediate_write_directory.name
  else:
    # Uses the provided write path, and creates a null context manager as a
    # stand-in for TemporaryDirectory.
    intermediate_write_directory = contextlib.nullcontext()
    path = intermediate_write_path
  with intermediate_write_directory:
    tfdf_model = _build_tfdf_model(sklearn_model, path)
  # The resultant tfdf model only receives the features that are used
  # to split samples in nodes in the trees as input. But we want to pass the
  # full design matrix as an input to match the scikit-learn API, thus we
  # create another tf.keras.Model with the desired call signature.
  template_input = tf.keras.Input(shape=(sklearn_model.n_features_in_,))
  # Extracts the indices of the features that are used by the TFDF model.
  # The features have names with the format "feature_<index-of-feature>".
  feature_names = tfdf_model.signatures[
      "serving_default"].structured_input_signature[1].keys()
  template_output = tfdf_model(
      {i: template_input[:, int(i.split("_")[1])] for i in feature_names})
  return tf.keras.Model(inputs=template_input, outputs=template_output)


@functools.singledispatch
def _build_tfdf_model(
    sklearn_model: ScikitLearnModel,
    path: os.PathLike,
) -> tf.keras.Model:
  """Builds a TFDF model from the given scikit-learn model."""
  raise NotImplementedError(
      f"Can't build a TFDF model for {type(sklearn_model)}")


@_build_tfdf_model.register(tree.DecisionTreeRegressor)
@_build_tfdf_model.register(tree.ExtraTreeRegressor)
def _(sklearn_model: ScikitLearnTree, path: os.PathLike) -> tf.keras.Model:
  """Converts a single scikit-learn regression tree to a TFDF model."""
  # The label argument is unused when the model is loaded, so we pass a
  # placeholder.
  objective = tfdf.py_tree.objective.RegressionObjective(label="label")
  pytree = convert_sklearn_tree_to_tfdf_pytree(sklearn_model)
  cart_builder = tfdf.builder.CARTBuilder(path=path, objective=objective)
  cart_builder.add_tree(pytree)
  cart_builder.close()
  return tf.keras.models.load_model(path)


@_build_tfdf_model.register(tree.DecisionTreeClassifier)
@_build_tfdf_model.register(tree.ExtraTreeClassifier)
def _(sklearn_model: ScikitLearnTree, path: os.PathLike) -> tf.keras.Model:
  """Converts a single scikit-learn classification tree to a TFDF model."""
  objective = tfdf.py_tree.objective.ClassificationObjective(
      label="label",
      # TF doesnt accept classes that aren't bytes or unicode,
      # so we convert the classes into strings in case they are not.
      classes=[str(c) for c in sklearn_model.classes_],
  )
  pytree = convert_sklearn_tree_to_tfdf_pytree(sklearn_model)
  cart_builder = tfdf.builder.CARTBuilder(path=path, objective=objective)
  cart_builder.add_tree(pytree)
  cart_builder.close()
  return tf.keras.models.load_model(path)


@_build_tfdf_model.register(ensemble.ExtraTreesRegressor)
@_build_tfdf_model.register(ensemble.RandomForestRegressor)
def _(
    sklearn_model: Union[ensemble.ExtraTreesRegressor, ensemble.RandomForestRegressor],
    path: os.PathLike,
) -> tf.keras.Model:
  """Converts a forest regression model into a TFDF model."""
  objective = tfdf.py_tree.objective.RegressionObjective(label="label")
  rf_builder = tfdf.builder.RandomForestBuilder(path=path, objective=objective)
  for single_tree in sklearn_model.estimators_:
    rf_builder.add_tree(convert_sklearn_tree_to_tfdf_pytree(single_tree))
  rf_builder.close()
  return tf.keras.models.load_model(path)


@_build_tfdf_model.register(ensemble.ExtraTreesClassifier)
@_build_tfdf_model.register(ensemble.RandomForestClassifier)
def _(
    sklearn_model: Union[ensemble.ExtraTreesClassifier, ensemble.RandomForestClassifier],
    path: os.PathLike,
) -> tf.keras.Model:
  """Converts a forest classification model into a TFDF model."""
  objective = tfdf.py_tree.objective.ClassificationObjective(
      label="label",
      classes=[str(c) for c in sklearn_model.classes_],
  )
  rf_builder = tfdf.builder.RandomForestBuilder(path=path, objective=objective)
  for single_tree in sklearn_model.estimators_:
    rf_builder.add_tree(convert_sklearn_tree_to_tfdf_pytree(single_tree))
  rf_builder.close()
  return tf.keras.models.load_model(path)


@_build_tfdf_model.register(ensemble.GradientBoostingRegressor)
def _(
    sklearn_model: ensemble.GradientBoostingRegressor,
    path: os.PathLike,
) -> tf.keras.Model:
  """Converts a gradient boosting regression model into a TFDF model."""
  if isinstance(sklearn_model.init_, dummy.DummyRegressor):
    # If the initial estimator is a DummyRegressor, then it predicts a constant
    # which can be passed to GradientBoostedTreeBuilder as a bias.
    init_pytree = None
    bias = sklearn_model.init_.constant_[0][0]
  elif isinstance(sklearn_model.init_, tree.DecisionTreeRegressor):
    # If the initial estimator is a DecisionTreeRegressor, we add it as the
    # first tree in the ensemble and set the bias to zero. We could also support
    # other tree-based initial estimators (e.g. RandomForest), but this seems
    # like a niche enough use case that we don't for the moment.
    init_pytree = convert_sklearn_tree_to_tfdf_pytree(sklearn_model.init_)
    bias = 0.0
  elif sklearn_model.init_ == "zero":
    init_pytree = None
    bias = 0.0
  else:
    raise ValueError("The initial estimator must be either a DummyRegressor"
                     "or a DecisionTreeRegressor, but got"
                     f"{type(sklearn_model.init_)}.")

  gbt_builder = tfdf.builder.GradientBoostedTreeBuilder(
      path=path,
      objective=tfdf.py_tree.objective.RegressionObjective(label="label"),
      bias=bias,
  )
  if init_pytree:
    gbt_builder.add_tree(init_pytree)

  for weak_learner in sklearn_model.estimators_.ravel():
    gbt_builder.add_tree(convert_sklearn_tree_to_tfdf_pytree(
        weak_learner,
        weight=sklearn_model.learning_rate,
    ))
  gbt_builder.close()
  return tf.keras.models.load_model(path)


def convert_sklearn_tree_to_tfdf_pytree(
    sklearn_tree: ScikitLearnTree,
    weight: Optional[float] = None,
) -> tfdf.py_tree.tree.Tree:
  """Converts a scikit-learn decision tree into a TFDF pytree.

  Args:
    sklearn_tree: a scikit-learn decision tree.
    weight: an optional weight to apply to the values of the leaves in the tree.
      This is intended for use when converting gradient boosted tree models.

  Returns:
    a TFDF pytree that has the same structure as the scikit-learn tree.
  """
  try:
    sklearn_tree_data = sklearn_tree.tree_.__getstate__()
  except AttributeError as e:
    raise ValueError(
        "Scikit-Learn model must be fit to data before converting.") from e

  field_names = sklearn_tree_data["nodes"].dtype.names
  task_type = _get_sklearn_tree_task_type(sklearn_tree)
  if weight and task_type is TaskType.SINGLE_LABEL_CLASSIFICATION:
    raise ValueError("weight should not be passed for classification trees.")

  nodes = []
  for node_properties, target_value in zip(
      sklearn_tree_data["nodes"],
      sklearn_tree_data["values"],
  ):
    node = {
        field_name: field_value
        for field_name, field_value in zip(field_names, node_properties)
    }
    if task_type is TaskType.SCALAR_REGRESSION:
      scaling_factor = weight if weight else 1.0
      node["value"] = tfdf.py_tree.value.RegressionValue(target_value[0][0] *
                                                         scaling_factor)
    elif task_type is TaskType.SINGLE_LABEL_CLASSIFICATION:
      # Normalise to probabilities if we have a classification tree.
      probabilities = list(target_value[0] / target_value[0].sum())
      node["value"] = tfdf.py_tree.value.ProbabilityValue(probabilities)
    else:
      raise ValueError(
          "Only scalar regression and single-label classification are "
          "supported.")
    nodes.append(node)

  root_node = _convert_sklearn_node_to_tfdf_node(
      # The root node has index zero.
      node_index=0,
      nodes=nodes,
  )
  return tfdf.py_tree.tree.Tree(root_node)


def _get_sklearn_tree_task_type(sklearn_tree: ScikitLearnTree) -> TaskType:
  """Finds the task type of a scikit learn tree."""
  if hasattr(sklearn_tree, "n_classes_") and sklearn_tree.n_outputs_ == 1:
    return TaskType.SINGLE_LABEL_CLASSIFICATION
  elif sklearn_tree.n_outputs_ == 1:
    return TaskType.SCALAR_REGRESSION
  else:
    return TaskType.UNKNOWN


def _convert_sklearn_node_to_tfdf_node(
    node_index: int,
    nodes: List[Dict[str, Any]],
) -> tfdf.py_tree.node.AbstractNode:
  """Converts a node within a scikit-learn tree into a TFDF node."""
  if node_index == -1:
    return None

  node = nodes[node_index]
  neg_child = _convert_sklearn_node_to_tfdf_node(
      node_index=node["left_child"],
      nodes=nodes,
  )
  pos_child = _convert_sklearn_node_to_tfdf_node(
      node_index=node["right_child"],
      nodes=nodes,
  )
  if pos_child:
    feature = tfdf.py_tree.dataspec.SimpleColumnSpec(
        name=f"feature_{node['feature']}",
        # In sklearn, all fields must be numerical.
        type=tfdf.py_tree.dataspec.ColumnType.NUMERICAL,
        col_idx=node["feature"],
    )
    return tfdf.py_tree.node.NonLeafNode(
        condition=tfdf.py_tree.condition.NumericalHigherThanCondition(
            feature=feature,
            threshold=node["threshold"],
            missing_evaluation=False,
        ),
        pos_child=pos_child,
        neg_child=neg_child,
    )
  else:
    return tfdf.py_tree.node.LeafNode(value=node["value"])
