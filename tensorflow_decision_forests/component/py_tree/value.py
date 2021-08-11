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

"""The value/prediction/output of a leaf node.

Non-leaf nodes can also have a value for debugging or model interpretation.
"""

import abc
import math
from typing import List, Optional

import numpy as np
import six

from yggdrasil_decision_forests.model.decision_tree import decision_tree_pb2


@six.add_metaclass(abc.ABCMeta)
class AbstractValue(object):
  """A generic value/prediction/output."""
  pass


class ProbabilityValue(AbstractValue):
  """A probability distribution value.

  Used for classification trees.

  Attrs:
    probability: An array of probability of the label classes i.e. the i-th
      value is the probability of the "label_value_idx_to_value(..., i)" class.
      Note that the first value is reserved for the Out-of-vocabulary
    num_examples: Number of example in the node.
  """

  def __init__(self,
               probability: List[float],
               num_examples: Optional[float] = 1.0):
    self._probability = probability
    self._num_examples = num_examples

  @property
  def probability(self):
    return self._probability

  @property
  def num_examples(self):
    return self._num_examples

  def __repr__(self):
    return f"ProbabilityValue({self._probability},n={self._num_examples})"

  def __eq__(self, other):
    if not isinstance(other, ProbabilityValue):
      return False
    return (self._probability == other._probability and
            self._num_examples == other._num_examples)


class RegressionValue(AbstractValue):
  """The regression value of a regressive tree.

  Can also be used in gradient-boosted-trees for classification and ranking.

  Attrs:
    value: Value of the tree. The semantic depends on the tree: For Random
      Forests, this value is a regressive value (in the same unit as the label).
      For classification and ranking GBDTs, this value is a loggit.
    standard_deviation: Optional standard deviation attached to the value.
    num_examples: Number of example in the node.
  """

  def __init__(self,
               value: float,
               num_examples: Optional[float] = 1.0,
               standard_deviation: Optional[float] = None):
    self._value = value
    self._standard_deviation = standard_deviation
    self._num_examples = num_examples

  @property
  def value(self):
    return self._value

  @property
  def standard_deviation(self):
    return self._standard_deviation

  @property
  def num_examples(self):
    return self._num_examples

  def __repr__(self):
    text = f"RegressionValue(value={self._value}"
    if self._standard_deviation is not None:
      text += f",sd={self._standard_deviation}"
    text += f",n={self._num_examples})"
    return text

  def __eq__(self, other):
    if not isinstance(other, RegressionValue):
      return False
    return (self._value == other._value and
            self._standard_deviation == other._standard_deviation and
            self._num_examples == other._num_examples)


def core_value_to_value(
    core_node: decision_tree_pb2.Node) -> Optional[AbstractValue]:
  """Converts a core value (proto format) into a python value."""

  if core_node.HasField("classifier"):
    dist = core_node.classifier.distribution
    probabilities = np.array(dist.counts[1:]) / dist.sum
    return ProbabilityValue(probabilities.tolist(), dist.sum)

  if core_node.HasField("regressor"):
    dist = core_node.regressor.distribution
    standard_deviation = None
    if dist.HasField("sum_squares") and dist.count > 0:
      variance = (
          dist.sum_squares / dist.count - (dist.sum * dist.sum) /
          (dist.count * dist.count))
      if variance >= 0:
        standard_deviation = math.sqrt(variance)
    return RegressionValue(core_node.regressor.top_value, dist.count,
                           standard_deviation)

  return None


def set_core_node(value: AbstractValue, core_node: decision_tree_pb2.Node):
  """Sets a core node (proto format) from a python value."""

  if isinstance(value, ProbabilityValue):
    dist = core_node.classifier.distribution
    dist.sum = value.num_examples
    dist.counts[:] = np.array([0.0] + value.probability) * dist.sum
    core_node.classifier.top_value = np.argmax(dist.counts)

  elif isinstance(value, RegressionValue):
    core_node.regressor.top_value = value.value
    if value.standard_deviation is not None:
      dist = core_node.regressor.distribution
      dist.count = value.num_examples
      dist.sum = 0
      dist.sum_squares = value.standard_deviation**2 * value.num_examples

  else:
    raise NotImplementedError("No supported value type")
