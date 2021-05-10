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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from absl import logging
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_decision_forests.component.inspector import blob_sequence
from yggdrasil_decision_forests.model.decision_tree import decision_tree_pb2


def data_root_path() -> str:
  return ""


def test_data_path() -> str:
  return os.path.join(data_root_path(),
                      "external/ydf/yggdrasil_decision_forests/test_data")


def tmp_path() -> str:
  return flags.FLAGS.test_tmpdir


def test_model_directory() -> str:
  return os.path.join(test_data_path(), "model")


class BlogSequenceTest(parameterized.TestCase, tf.test.TestCase):

  def test_base(self):
    path = os.path.join(tmp_path(), "tmp.bs")

    writer = blob_sequence.Writer(path)
    writer.write(b"HELLO")
    writer.write(b"WORLD")
    writer.close()

    reader = blob_sequence.Reader(path)
    self.assertEqual([b"HELLO", b"WORLD"], list(reader))

  def test_node(self):
    path = os.path.join(test_model_directory(), "adult_binary_class_rf",
                        "nodes-00000-of-00001")
    num_nodes = 0
    for serialized_node in blob_sequence.Reader(path):
      node = decision_tree_pb2.Node.FromString(serialized_node)
      if num_nodes <= 2:
        logging.info("Node: %s", node)
      num_nodes += 1
    # Matches the number of nodes in "description.txt".
    self.assertEqual(num_nodes, 125578)


if __name__ == "__main__":
  tf.test.main()
