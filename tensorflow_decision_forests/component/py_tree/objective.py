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

"""Definition of a model objective.

An objective contains a task (e.g. classification), a label column name, and
objective specific meta-data e.g. the number of classes for a classification
objective.
"""

import abc
from typing import List, Optional

import six
from yggdrasil_decision_forests.model import abstract_model_pb2

Task = abstract_model_pb2.Task


@six.add_metaclass(abc.ABCMeta)
class AbstractObjective(object):
  """Abstract objective."""

  def __init__(self, label: str):
    if not label:
      raise ValueError("The label cannot be the empty string")
    self._label = label

  @property
  def label(self):
    return self._label

  @property
  @abc.abstractmethod
  def task(self) -> Task:
    """Task value."""

    raise NotImplementedError()


class ClassificationObjective(AbstractObjective):
  """Objective for classification."""

  def __init__(self,
               label: str,
               classes: Optional[List[str]] = None,
               num_classes: Optional[int] = None):
    """Create the objective.

    Either `classes` or `num_classes` should be provided. If both are provided,
    `classes` should contain `num_classes` elements.

    If only `num_classes` is provided, the label is assumed to be an integer in
    [0, num_classes).

    Args:
      label: Objective label name.
      classes: List of possible class values.
      num_classes: Number of classes.
    """

    super(ClassificationObjective, self).__init__(label)

    if classes is None and num_classes is None:
      raise ValueError(
          "At least one of classes or num_classes should be provided")

    if classes is not None and num_classes is not None:
      if len(classes) != num_classes:
        raise ValueError("If both num_classes and classes are provided, "
                         "classes should contain num_classes elements.")
    elif classes is not None:
      num_classes = len(classes)

    if num_classes < 2:  # pytype: disable=unsupported-operands
      raise ValueError("The number of unique classes should be at least 2 i.e."
                       " binary classification")

    self._classes = classes
    self._num_classes = num_classes

  @property
  def num_classes(self) -> int:
    return self._num_classes

  @property
  def classes(self) -> Optional[List[str]]:
    return self._classes

  @property
  def has_integer_labels(self) -> bool:
    return self._classes is None

  @property
  def task(self) -> "Task":
    return Task.CLASSIFICATION

  def __repr__(self):
    return (f"Classification(label={self.label}, class={self._classes}, "
            f"num_classes={self._num_classes})")

  def __eq__(self, other):
    if not isinstance(other, ClassificationObjective):
      return False
    return (self.label == other.label and self._classes == other._classes and
            self._num_classes == other._num_classes)


class RegressionObjective(AbstractObjective):
  """Objective for regression."""

  @property
  def task(self) -> Task:
    return Task.REGRESSION

  def __repr__(self):
    return f"Regression(label={self.label})"

  def __eq__(self, other):
    if not isinstance(other, RegressionObjective):
      return False
    return self.label == other.label


class RankingObjective(AbstractObjective):
  """Objective for ranking."""

  def __init__(self, label: str, group: str):
    super(RankingObjective, self).__init__(label)
    self._group = group

  @property
  def group(self) -> str:
    return self._group

  @property
  def task(self) -> Task:
    return Task.RANKING

  def __repr__(self):
    return f"Ranking(label={self.label}, group={self._group}"

  def __eq__(self, other):
    if not isinstance(other, RankingObjective):
      return False
    return self.label == other.label and self._group == other._group


class AbstractUpliftObjective(AbstractObjective):
  """Objective for Uplift."""

  def __init__(self, label: str, treatment: str):
    super(AbstractUpliftObjective, self).__init__(label)
    self._treatment = treatment

  @property
  def treatment(self) -> str:
    return self._treatment

  def __eq__(self, other):
    if not isinstance(other, AbstractUpliftObjective):
      return False
    return (
        self.label == other.label
        and self._treatment == other._treatment
        and self.task == other.task
    )


class CategoricalUpliftObjective(AbstractUpliftObjective):
  """Objective for Categorical Uplift."""

  @property
  def task(self) -> Task:
    return Task.CATEGORICAL_UPLIFT

  def __repr__(self):
    return f"CategoricalUplift(label={self.label}, treatment={self._treatment})"


class NumericalUpliftObjective(AbstractUpliftObjective):
  """Objective for Numerical Uplift."""

  @property
  def task(self) -> Task:
    return Task.NUMERICAL_UPLIFT

  def __repr__(self):
    return f"NumericalUplift(label={self.label}, treatment={self._treatment})"
