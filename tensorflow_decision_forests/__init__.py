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

# -*- coding: utf-8 -*-
"""User entry point for the TensorFlow Decision Forest API.

Basic usage:

```
# Imports
import tensorflow_decision_forests as tfdf
import pandas as pd
from wurlitzer import sys_pipes

# Load a dataset into a Pandas Dataframe.
dataset_df = pd.read_csv("/tmp/penguins.csv")

# Display the first 3 examples.
dataset_df.head(3)

# Convert the Pandas dataframe to a tf dataset
tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset_df,label="species")

model = tfdf.keras.RandomForestModel()
with sys_pipes():
  model.fit(tf_dataset)
# Note: The `sys_pipes` part is to display logs during training.

# Evaluate model.
model.compile(metrics=["accuracy"])
model.evaluate(...test_dataset...)

# Save model.
model.save("/tmp/my_saved_model")

# ...

# Load a model: it loads as a generic keras model.
loaded_model = tf_keras.models.load_model("/tmp/my_saved_model")
```
"""

__version__ = "1.12.0"
__author__ = "Mathieu Guillame-Bert"

compatible_tf_versions = ["2.20.0"]
__git_version__ = "HEAD"  # Modify for release build.

from tensorflow_decision_forests.tensorflow import check_version

check_version.check_version(__version__, compatible_tf_versions)

from tensorflow_decision_forests import keras
from tensorflow_decision_forests.component import py_tree
from tensorflow_decision_forests.component.builder import builder
from tensorflow_decision_forests.component.inspector import inspector
from tensorflow_decision_forests.component.model_plotter import model_plotter
from tensorflow_decision_forests.component.tuner import tuner

if __name__ == "__main__":

  import os
  import sys
  import io

  def _is_direct_output(stream=sys.stdout):
    """Checks if output stream redirects to the shell/console directly."""

    if stream.isatty():
      return True
    if isinstance(stream, io.TextIOWrapper):
      return _is_direct_output(stream.buffer)
    if isinstance(stream, io.BufferedWriter):
      return _is_direct_output(stream.raw)
    if isinstance(stream, io.FileIO):
      return stream.fileno() in [1, 2]
    return False

  # Only print the welcome message if TFDF_DISABLE_WELCOME_MESSAGE is not set
  # and if user has not already imported YDF.
  if (
      os.getenv("TFDF_DISABLE_WELCOME_MESSAGE") is None
      and "ydf" not in sys.modules
  ):

    if not _is_direct_output():  # Check if executed in a Notebook

      import IPython

      IPython.display.display(IPython.display.HTML("""
<p style="margin:0px;">🌲 Try <a href="https://ydf.readthedocs.io/en/latest/" target="_blank">YDF</a>, the successor of
    <a href="https://www.tensorflow.org/decision_forests" target="_blank">TensorFlow
        Decision Forests</a> using the same algorithms but with more features and faster
    training!
</p>
<div style="display: flex; flex-wrap: wrap; margin:5px;max-width: 880px;">
    <div style="flex: 1; border-radius: 10px; background-color: F0F0F0; padding: 5px;">
        <p
            style="font-weight: bold; margin:0px;text-align: center;border-bottom: 1px solid #C0C0C0;margin-bottom: 4px;">
            Old code</p>
        <pre style="overflow-wrap: anywhere; overflow: auto; margin:0px;font-size: 9pt;">
import tensorflow_decision_forests as tfdf

tf_ds = tfdf.keras.pd_dataframe_to_tf_dataset(ds, label="l")
model = tfdf.keras.RandomForestModel(label="l")
model.fit(tf_ds)
</pre>
    </div>
    <div style="width: 5px;"></div>
    <div style="flex: 1; border-radius: 10px; background-color: F0F0F0; padding: 5px;">
        <p
            style="font-weight: bold; margin:0px;text-align: center;border-bottom: 1px solid #C0C0C0;margin-bottom: 4px;">
            New code</p>
        <pre style="overflow-wrap: anywhere; overflow: auto; margin:0px;font-size: 9pt;">
import ydf

model = ydf.RandomForestLearner(label="l").train(ds)
</pre>
    </div>
</div>
<p style="margin:0px;font-size: 9pt;">(Learn more in the <a
        href="https://ydf.readthedocs.io/en/latest/tutorial/migrating_to_ydf/" target="_blank">migration
        guide</a>)</p>
  """))

    else:
      import termcolor
      print(
          termcolor.colored("🌲 Try ", "green"),
          termcolor.colored("https://ydf.readthedocs.io", "blue"),
          termcolor.colored(
              ", the successor of TensorFlow Decision Forests with more"
              " features and faster training!",
              "green",
          ),
          sep="",
          file=sys.stderr,
      )
      pass
