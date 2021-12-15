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

from tensorflow_decision_forests.tensorflow import check_version
import tensorflow as tf
from tensorflow.python.platform import resource_loader
import sys

try:
  ops = tf.load_op_library(resource_loader.get_path_to_datafile("inference.so"))
except Exception as e:
  check_version.info_fail_to_load_custom_op(e, "inference.so")
  raise e

# Importing all the symbols.
module = sys.modules[__name__]
for name, value in ops.__dict__.items():
  if "__" in name:
    continue
  setattr(module, name, value)
