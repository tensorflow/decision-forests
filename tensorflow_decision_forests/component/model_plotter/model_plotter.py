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

"""Plotting of decision forest models.

Standalone usage:
  import tensorflow_decision_forests as tfdf
  model = tfdf.keras.CartModel().
  model.fit(...)

  with open("plot.html", "w") as f:
    f.write(tfdf.model_plotter.plot_model(model))

Colab usage:

  import tensorflow_decision_forests as tfdf
  model = tfdf.keras.CartModel().
  model.fit(...)

  tfdf.model_plotter.plot_model_in_colab(model)
"""

import json
import string
from typing import Dict, Any, Optional, NamedTuple
import uuid

import tensorflow as tf

from tensorflow_decision_forests.component.inspector import inspector as inspector_lib
from tensorflow_decision_forests.component.py_tree import condition as condition_lib
from tensorflow_decision_forests.component.py_tree import node as node_lib
from tensorflow_decision_forests.component.py_tree import tree as tree_lib
from tensorflow_decision_forests.component.py_tree import value as value_lib

# The InferenceCoreModel" is defined in /keras/core_inference.py.
InferenceCoreModel = Any


class DisplayOptions(NamedTuple):
  """Display options.

  All the values are expressed in pixel.
  """

  # Margin around the entire plot.
  margin: Optional[float] = 10

  # Size of a tree node.
  node_x_size: Optional[float] = 160
  node_y_size: Optional[float] = 12 * 2 + 4

  # Space between tree nodes.
  node_x_offset: Optional[float] = 160 + 20
  node_y_offset: Optional[float] = 12 * 2 + 4 + 5

  # Text size.
  font_size: Optional[float] = 10

  # Rounding effect of the edges.
  # This value is the distance (in pixel) of the Bezier control anchor from
  # the source point.
  edge_rounding: Optional[float] = 20

  # Padding inside nodes.
  node_padding: Optional[float] = 2

  # Show a bb box around the plot. For debug only.
  show_plot_bounding_box: Optional[bool] = False


def plot_model_in_colab(model: InferenceCoreModel, **kwargs):
  """Plots a model structure in colab.

  See "plot_model" for the available options.

  Args:
    model: The model to plot.
    **kwargs: Arguments passed to "plot_model".

  Returns:
    A Colab HTML element showing the model.
  """

  from IPython.display import HTML  # pylint: disable=g-import-not-at-top # pytype: disable=import-error
  return HTML(plot_model(model, **kwargs))


def plot_model(model: InferenceCoreModel,
               tree_idx: Optional[int] = 0,
               max_depth: Optional[int] = 3):
  """Plots the model structure and its correlation with a dataset.

  Args:
    model: The model to plot.
    tree_idx: If the model is composed of multiple trees, only plot this tree.
    max_depth: Maximum depth to plot. If None, does not limit the depth.

  Returns:
    Html representation of the tree.
  """

  # What method is available in the model (depend on how keras serialize it).
  has_make_inspector = callable(getattr(model, "make_inspector", None))
  has_yggdrasil_model_path_tensor = callable(
      getattr(model, "yggdrasil_model_path_tensor", None))

  if has_make_inspector:
    inspector = model.make_inspector()

  elif has_yggdrasil_model_path_tensor:
    path = model.yggdrasil_model_path_tensor().numpy().decode("utf-8")
    inspector = inspector_lib.make_inspector(path)

  else:
    raise ValueError(
        "The model does not have a 'make_inspector' or"
        "'get_yggdrasil_model_path' method. Make sure this is a valid"
        "TensorFlow Decision Forest model")

  tree = inspector.extract_tree(tree_idx)
  return plot_tree(tree=tree, max_depth=max_depth)


def plot_tree(tree: tree_lib.Tree,
              max_depth: Optional[int] = None,
              display_options: Optional[DisplayOptions] = None) -> str:
  """Plots a decision tree.

  Args:
    tree: A decision tree.
    max_depth: Maximum plotting depth. Makes the plot more readable in case of
      large trees.
    display_options: Dictionary of display options.

  Returns:
    The html content displaying the tree.
  """

  # Plotting library.
  plotter_js_path = tf.compat.v1.resource_loader.get_path_to_datafile(
      "plotter.js")
  with open(plotter_js_path) as f:
    plotter_js_content = f.read()

  container_id = "tree_plot_" + uuid.uuid4().hex

  # Converts the tree into its json representation.
  json_tree = _tree_to_json(tree, max_depth)

  # Display options.
  if display_options is None:
    display_options = DisplayOptions()
  options = dict(display_options._asdict())

  if tree.label_classes is not None:
    options["labels"] = json.dumps(tree.label_classes)

  html_content = string.Template("""
<script src="https://d3js.org/d3.v6.min.js"></script>
<div id="${container_id}"></div>
<script>
${plotter_js_content}
display_tree(${options}, ${json_tree_content}, "#${container_id}")
</script>
""").substitute(
    options=json.dumps(options),
    plotter_js_content=plotter_js_content,
    container_id=container_id,
    json_tree_content=json.dumps(json_tree))

  return html_content


def _tree_to_json(src: tree_lib.Tree,
                  max_depth: Optional[int]) -> Dict[str, Any]:
  """Converts a tree into a json object compatible with the plotter."""

  if src.root is None:
    return {}

  return _node_to_json(src.root, max_depth, 0)


def _node_to_json(src: node_lib.AbstractNode, max_depth: Optional[int],
                  depth: int) -> Dict[str, Any]:
  """Converts a node into a json object compatible with the plotter."""

  dst = {}
  if isinstance(src, node_lib.LeafNode):
    dst["value"] = _value_to_json(src.value)

  elif isinstance(src, node_lib.NonLeafNode):
    if src.value is not None:
      dst["value"] = _value_to_json(src.value)

    dst["condition"] = _condition_to_json(src.condition)
    if max_depth is not None and depth < max_depth:
      dst["children"] = [
          _node_to_json(src.pos_child, max_depth, depth + 1),
          _node_to_json(src.neg_child, max_depth, depth + 1)
      ]

  else:
    raise ValueError(f"Non supported node type {src}")

  return dst


def _condition_to_json(src: condition_lib.AbstractCondition) -> Dict[str, Any]:
  """Converts a condition into a json object compatible with the plotter."""

  if isinstance(src, condition_lib.IsMissingInCondition):
    return {"type": "IS_MISSING", "attribute": src.feature.name}

  if isinstance(src, condition_lib.IsTrueCondition):
    return {"type": "IS_TRUE", "attribute": src.feature.name}

  if isinstance(src, condition_lib.NumericalHigherThanCondition):
    return {
        "type": "NUMERICAL_IS_HIGHER_THAN",
        "attribute": src.feature.name,
        "threshold": src.threshold
    }

  if isinstance(src, condition_lib.CategoricalIsInCondition):
    return {
        "type": "CATEGORICAL_IS_IN",
        "attribute": src.feature.name,
        "mask": src.mask
    }

  if isinstance(src, condition_lib.CategoricalSetContainsCondition):
    return {
        "type": "CATEGORICAL_SET_CONTAINS",
        "attribute": src.feature.name,
        "mask": src.mask
    }

  if isinstance(src, condition_lib.NumericalSparseObliqueCondition):
    return {
        "type": "NUMERICAL_SPARSE_OBLIQUE",
        "attributes": [f.name for f in src.features()],
        "weights": src.weights,
        "threshold": src.threshold
    }

  raise ValueError(f"Non supported condition type {src}")


def _value_to_json(src: value_lib.AbstractValue) -> Dict[str, Any]:
  """Converts a value into a json object compatible with the plotter."""

  if isinstance(src, value_lib.ProbabilityValue):
    return {
        "type": "PROBABILITY",
        "distribution": src.probability,
        "num_examples": src.num_examples
    }

  elif isinstance(src, value_lib.RegressionValue):
    value = {
        "type": "REGRESSION",
        "value": src.value,
        "num_examples": src.num_examples
    }
    if src.standard_deviation is not None:
      value["standard_deviation"] = src.standard_deviation
    return value

  raise ValueError(f"Non supported value type {src}")
