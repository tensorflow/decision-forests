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

"""Nodes (leaf and non-leafs) in a tree."""

import abc
from collections import defaultdict
from typing import Optional, List, Tuple, Dict

import six

from tensorflow_decision_forests.component.py_tree import condition as condition_lib
from tensorflow_decision_forests.component.py_tree import value as value_lib
from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.model.decision_tree import decision_tree_pb2

AbstractCondition = condition_lib.AbstractCondition
AbstractValue = value_lib.AbstractValue

# Number of spaces printed on the left side of nodes with pretty print.
_PRETTY_MARGIN = 4
# Length / number of characters (e.g. "-") in an edge with pretty print.
_PRETTY_EDGE_LENGTH = 4


class ConditionValueAndDefaultEvaluation(object):
  """Set of condition values and default evaluations per features.

  Attributes:
    numerical_higher_than: List of (threshold, default_eval) for the conditions
      of the shape "a>=t". Indexed by feature name.
  """

  def __init__(self):
    self._numerical_higher_than: Dict[str, List[Tuple[
        float, bool]]] = defaultdict(lambda: [])

  @property
  def numerical_higher_than(self):
    return self._numerical_higher_than


@six.add_metaclass(abc.ABCMeta)
class AbstractNode(object):
  """A decision tree node."""

  def collect_condition_parameter_and_default_evaluation(
      self, conditions: ConditionValueAndDefaultEvaluation):
    """Extracts the condition values and default evaluations."""
    pass

  @abc.abstractmethod
  def pretty(self, prefix: str, is_pos: Optional[bool], depth: int,
             max_depth: Optional[int]) -> str:
    """Returns a recursive readable textual representation of a node.

    Args:
      prefix: Prefix printed on the left side. Used to print the surrounding
        edges.
      is_pos: True/False if the node is a positive/negative child. None if the
        node is a root.
      depth: Depth of the node in the tree. There is no assuption of on the
        depth of a root.
      max_depth: Maximum depth for representation. Deeper nodes are skipped.

    Returns:
      A pretty-string representing the node and its children.
    """

    raise NotImplementedError()

  def __str__(self):
    # Prints a node and its descendants.
    return self.pretty("", None, 1, None)


class LeafNode(AbstractNode):
  """A leaf node i.e. the node containing a prediction/value/output."""

  def __init__(self, value: AbstractValue, leaf_idx: Optional[int] = None):
    self._value = value
    self._leaf_idx = leaf_idx

  def __repr__(self):
    return f"LeafNode(value={self._value}, idx={self._leaf_idx})"

  @property
  def value(self):
    return self._value

  @value.setter
  def value(self, value):
    self._value = value

  @property
  def leaf_idx(self) -> Optional[int]:
    """Index of the leaf in the tree in a depth first exploration."""

    return self._leaf_idx

  @leaf_idx.setter
  def leaf_idx(self, leaf_idx):
    self._leaf_idx = leaf_idx

  def pretty(self, prefix: str, is_pos: Optional[bool], depth: int,
             max_depth: Optional[int]) -> str:
    text = prefix + _pretty_local_prefix(is_pos) + str(self._value)
    if self._leaf_idx is not None:
      text += f" (idx={self._leaf_idx})"
    return text + "\n"


class NonLeafNode(AbstractNode):
  """A non-leaf node i.e.

  a node containing a split/condition.

  Attrs:
    condition: The binary condition of the node.
    pos_child: The child to visit when the condition is true.
    neg_child: The child to visit when the condition is false.
    value: The value/prediction/output of the node if it was a leaf. Not used
      during prediction.
  """

  def __init__(self,
               condition: AbstractCondition,
               pos_child: Optional[AbstractNode] = None,
               neg_child: Optional[AbstractNode] = None,
               value: Optional[AbstractValue] = None):

    self._condition = condition
    self._pos_child = pos_child
    self._neg_child = neg_child
    self._value = value

  @property
  def condition(self):
    return self._condition

  @condition.setter
  def condition(self, value):
    self._condition = value

  @property
  def pos_child(self):
    return self._pos_child

  @pos_child.setter
  def pos_child(self, value):
    self._pos_child = value

  @property
  def neg_child(self):
    return self._neg_child

  @neg_child.setter
  def neg_child(self, value):
    self._neg_child = value

  @property
  def value(self):
    return self._value

  @value.setter
  def value(self, value):
    self._value = value

  def collect_condition_parameter_and_default_evaluation(
      self, conditions: ConditionValueAndDefaultEvaluation):
    """Extracts the condition values and default evaluations."""

    if isinstance(self._condition, condition_lib.NumericalHigherThanCondition):
      conditions.numerical_higher_than[self._condition.feature.name].append(
          (self._condition.threshold, self._condition.missing_evaluation))

    if self._pos_child is not None:
      self._pos_child.collect_condition_parameter_and_default_evaluation(
          conditions)

    if self._neg_child is not None:
      self._neg_child.collect_condition_parameter_and_default_evaluation(
          conditions)

  def __repr__(self):
    # Note: Make sure to use `repr` instead of `str`.
    text = "NonLeafNode(condition=" + repr(self._condition)

    # Positive child.
    if self._pos_child is not None:
      text += f", pos_child={repr(self._pos_child)}"
    else:
      text += ", pos_child=None"

    # Negative child.
    if self._neg_child is not None:
      text += f", neg_child={repr(self._neg_child)}"
    else:
      text += ", neg_child=None"

    # Node value.
    if self._value is not None:
      text += f", value={repr(self._value)}"
    text += ")"
    return text

  def pretty(self, prefix: str, is_pos: Optional[bool], depth: int,
             max_depth: Optional[int]) -> str:

    # Prefix for the children of this node.
    children_prefix = prefix
    if is_pos is None:
      pass
    elif is_pos:
      children_prefix += " " * _PRETTY_MARGIN + "│" + " " * _PRETTY_EDGE_LENGTH
    elif not is_pos:
      children_prefix += " " * (_PRETTY_MARGIN + 1 + _PRETTY_EDGE_LENGTH)

    # Node's condition.
    text = prefix + _pretty_local_prefix(is_pos) + str(self._condition) + "\n"

    # Children of the node.
    if max_depth is not None and depth >= max_depth:
      if self._pos_child is not None or self._neg_child is not None:
        text += children_prefix + "...\n"
    else:
      if self._pos_child is not None:
        text += self._pos_child.pretty(children_prefix, True, depth + 1,
                                       max_depth)
      if self._neg_child is not None:
        text += self._neg_child.pretty(children_prefix, False, depth + 1,
                                       max_depth)
    return text


def _pretty_local_prefix(is_pos: Optional[bool]) -> str:
  """Prefix added in front of a node with pretty print.

  Args:
    is_pos: True/False if the node is a positive/negative child. None if the
      node is a root.

  Returns:
    The node prefix.
  """

  if is_pos is None:
    # Root node. No prefix.
    return ""
  elif is_pos:
    # Positive nodes are assumed to be printed before negative ones.
    return " " * _PRETTY_MARGIN + "├─(pos)─ "
  else:
    return " " * _PRETTY_MARGIN + "└─(neg)─ "


def core_node_to_node(
    core_node: decision_tree_pb2.Node,
    dataspec: data_spec_pb2.DataSpecification) -> AbstractNode:
  """Converts a core node (proto format) into a python node."""

  if core_node.HasField("condition"):
    # Non leaf
    return NonLeafNode(
        condition=condition_lib.core_condition_to_condition(
            core_node.condition, dataspec),
        value=value_lib.core_value_to_value(core_node))

  else:
    # Leaf
    value = value_lib.core_value_to_value(core_node)
    if value is None:
      raise ValueError("Leaf node should have a value")
    return LeafNode(value)


def node_to_core_node(
    node: AbstractNode,
    dataspec: data_spec_pb2.DataSpecification) -> decision_tree_pb2.Node:
  """Converts a python node into a core node (proto format)."""

  core_node = decision_tree_pb2.Node()
  if isinstance(node, NonLeafNode):
    condition_lib.set_core_node(node.condition, dataspec, core_node)
    if node.value is not None:
      value_lib.set_core_node(node.value, core_node)

  elif isinstance(node, LeafNode):
    value_lib.set_core_node(node.value, core_node)

  else:
    raise ValueError(
        f"Expecting a LeafNode or a NonLeafNode. Got {node} instead")

  return core_node
