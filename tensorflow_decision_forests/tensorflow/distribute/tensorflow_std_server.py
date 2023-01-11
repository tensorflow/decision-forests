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

"""Python worker for ParameterServerStrategy.

This binary is a distributed training worker. When using distributed training,
this worker (or the c++ worker defined in the same BUILD file) should run on
worker machines. The configuration of the worker and chief is done
though environement variable. See examples/distributed_training.py

When possible, use the c++ worker instead.
"""

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
import tensorflow_decision_forests as tfdf  # pylint: disable=unused-import

FLAGS = flags.FLAGS

flags.DEFINE_string("job_name", "worker", "")
flags.DEFINE_string("protocol", "grpc", "")
flags.DEFINE_integer("task_index", 0, "")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  logging.info("Starting worker")
  cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
  server = tf.distribute.Server(
      cluster_resolver.cluster_spec(),
      protocol=FLAGS.protocol,
      job_name=FLAGS.job_name,
      task_index=FLAGS.task_index)

  logging.info("Worker is running")
  server.join()

  logging.info("Shutting down worker")


if __name__ == "__main__":
  app.run(main)
