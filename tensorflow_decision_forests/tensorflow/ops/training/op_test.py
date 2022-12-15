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

"""Tests for op."""

import logging

import tensorflow as tf

from tensorflow_decision_forests.tensorflow.ops.training import op


class OpTest(tf.test.TestCase):

  def test_grpc_workers(self):
    port = op.SimpleMLCreateYDFGRPCWorker(key=1)
    logging.info("port: %d", port)

    port_again = op.SimpleMLCreateYDFGRPCWorker(key=1)
    logging.info("port_again: %d", port_again)

    self.assertEqual(port, port_again)

    port_other_server = op.SimpleMLCreateYDFGRPCWorker(key=2)
    logging.info("port_other_server: %d", port_other_server)

    self.assertNotEqual(port, port_other_server)

    op.SimpleMLStopYDFGRPCWorker(key=1)
    op.SimpleMLStopYDFGRPCWorker(key=2)


if __name__ == "__main__":
  tf.test.main()
