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

"""Decision Forest in a Keras Model.

Usage example:

```python
import tensorflow_decision_forests as tfdf
import pandas as pd

# Load the dataset in a Pandas dataframe.
train_df = pd.read_csv("project/train.csv")
test_df = pd.read_csv("project/test.csv")

# Convert the dataset into a TensorFlow dataset.
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="my_label")
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="my_label")

# Train the model.
model = tfdf.keras.RandomForestModel()
model.fit(train_ds)

# Evaluate the model on another dataset.
model.evaluate(test_ds)

# Show information about the model
model.summary()

# Export the model with the TF.SavedModel format.
model.save("/path/to/my/model")
```

"""

from typing import Callable, List

from tensorflow_decision_forests.keras import core
from tensorflow_decision_forests.keras import wrappers

# Utility classes
CoreModel = core.CoreModel
FeatureSemantic = core.FeatureSemantic
Task = core.Task
FeatureUsage = core.FeatureUsage
AdvancedArguments = core.AdvancedArguments

# Learning algorithm (called Models in Keras).


class RandomForestModel(wrappers.RandomForestModel):
  pass


class GradientBoostedTreesModel(wrappers.GradientBoostedTreesModel):
  pass


class CartModel(wrappers.CartModel):
  pass


class DistributedGradientBoostedTreesModel(
    wrappers.DistributedGradientBoostedTreesModel):
  pass


def get_all_models() -> List[Callable[[], CoreModel]]:
  """Gets the lists of all the available models."""
  return [
      RandomForestModel, GradientBoostedTreesModel, CartModel,
      DistributedGradientBoostedTreesModel
  ]


# Utilities
pd_dataframe_to_tf_dataset = core.pd_dataframe_to_tf_dataset
get_worker_idx_and_num_workers = core.get_worker_idx_and_num_workers
