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

"""Specification of the parameters of a tuner.

The "tuner" is a meta-learning algorithm that find the optimal hyperparameter
values of a base learner. "Tuner" is the TF-DF name for the YDF automatic
Hyperparameter optimizer V2. For example, a tuner can find the hyper-parameters
that maximize the accuracy of a GradientBoostedTreesModel model.

Usage example:

```
# Imports
import tensorflow_decision_forests as tfdf

# Load a dataset into a Pandas Dataframe.
dataset_df = pd.read_csv("/tmp/penguins.csv")

# Convert the Pandas dataframe to a tf dataset
tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset_df,label="species")

# Configure the tuner.
tuner = tfdf.tuner.RandomSearch(num_trials=20)
tuner.choice("num_candidate_attributes_ratio", [1.0, 0.8, 0.6])
tuner.choice("use_hessian_gain", [True, False])

local_search_space = tuner.choice("growing_strategy", ["LOCAL"])
local_search_space.choice("max_depth", [4, 5, 6, 7])

global_search_space = tuner.choice(
    "growing_strategy", ["BEST_FIRST_GLOBAL"], merge=True)
global_search_space.choice("max_num_nodes", [16, 32, 64, 128])

# Configure and train the model.
model = tfdf.keras.GradientBoostedTreesModel(num_trees=50, tuner=tuner)
model.fit(tf_dataset)
```
"""

from __future__ import annotations

from typing import List, Optional, Union, Any

from yggdrasil_decision_forests.learner import abstract_learner_pb2
from yggdrasil_decision_forests.learner.hyperparameters_optimizer import hyperparameters_optimizer_pb2
from yggdrasil_decision_forests.learner.hyperparameters_optimizer.optimizers import random_pb2
from yggdrasil_decision_forests.model import hyperparameter_pb2

TrainConfig = abstract_learner_pb2.TrainingConfig
HPOptProto = hyperparameters_optimizer_pb2.HyperParametersOptimizerLearnerTrainingConfig

Fields = Any  # repeated Field fields


class SearchSpace(object):
  """Set of hyperparameter and their respective possible values.

  The user is not expected to create a "SearchSpace" object directly. Instead,
  SearchSpace object are instantiated by tuners.
  """

  def __init__(
      self,
      fields: Fields,
      parent_values: Optional[
          hyperparameter_pb2.HyperParameterSpace.DiscreteCandidates] = None):
    self._fields = fields
    self._parent_values = parent_values

  def choice(self,
             key: str,
             values: Union[List[int], List[float], List[str], List[bool]],
             merge: bool = False) -> "SearchSpace":
    """Adds a hyperparameter with a list of possible values.

    Args:
      key: Name of the hyper-parameter.
      values: List of possible value for the hyperparameter.
      merge: If false (default), raises an error if the hyper-parameter already
        exist. If true, adds values to the parameter if it already exist.

    Returns:
      The conditional SearchSpace corresponding to the values in "values".
    """

    if not values:
      raise ValueError("The list of values is empty")

    field = self._find_field(key)
    if field is None:
      if merge:
        raise ValueError(
            f"Merge=true but the field {key} does not already exist")
      field = self._fields.add(name=key)
      if self._parent_values:
        field.parent_discrete_values.MergeFrom(self._parent_values)
    else:
      if not merge:
        raise ValueError(f"The field {key} already exist")

    dst_values = hyperparameter_pb2.HyperParameterSpace.DiscreteCandidates()
    for value in values:
      dst_value = dst_values.possible_values.add()
      if isinstance(value, bool):
        dst_value.categorical = "true" if value else "false"
      elif isinstance(value, int):
        dst_value.integer = value
      elif isinstance(value, float):
        dst_value.real = value
      elif isinstance(value, str):
        dst_value.categorical = value
      else:
        raise ValueError(f"Not supported value type: {value}:{type(value)}")

    field.discrete_candidates.possible_values.extend(
        dst_values.possible_values[:])

    return SearchSpace(field.children, parent_values=dst_values)

  def _find_field(
      self, key: str) -> Optional[hyperparameter_pb2.HyperParameterSpace.Field]:
    """Gets the existing hyperparameter with this name."""

    for field in self._fields:
      if field.name == key:
        return field
    return None


class Tuner(object):
  """Abstract tuner class.

  The user is expected to use one of its instances e.g. RandomSearch.
  """

  def __init__(self):
    self._train_config = TrainConfig(learner="HYPERPARAMETER_OPTIMIZER")

  def train_config(self) -> TrainConfig:
    """YDF training configuration for the Hyperparameter optimizer."""

    return self._train_config

  def set_base_learner(self, learner: str) -> None:
    """Sets the base learner key."""

    self._optimizer_proto().base_learner.learner = learner

  def choice(self,
             key: str,
             values: Union[List[int], List[float], List[str], List[bool]],
             merge: bool = False) -> SearchSpace:
    """Adds a hyperparameter with a list of possible values.

    Args:
      key: Name of the hyper-parameter.
      values: List of possible value for the hyperparameter.
      merge: If false (default), raises an error if the hyper-parameter already
        exist. If true, adds values to the parameter if it already exist.

    Returns:
      The conditional SearchSpace corresponding to the values in "values".
    """

    sp = SearchSpace(self._optimizer_proto().search_space.fields)
    return sp.choice(key, values, merge)

  def _optimizer_proto(self) -> HPOptProto:
    return self._train_config.Extensions[
        hyperparameters_optimizer_pb2.hyperparameters_optimizer_config]


class RandomSearch(Tuner):
  """Tuner using random hyperparameter values.

  The candidate hyper-parameter can be evaluated independently and in parallel.

  Attributes:
    num_trials: Number of random hyperparameter values to evaluate.
  """

  def __init__(self, num_trials: int = 100):
    super(RandomSearch, self).__init__()
    self._optimizer_proto().optimizer.optimizer_key = "RANDOM"
    self._random_optimizer_proto().num_trials = num_trials

  def _random_optimizer_proto(self) -> random_pb2.RandomOptimizerConfig:
    return self._optimizer_proto().optimizer.Extensions[random_pb2.random]
