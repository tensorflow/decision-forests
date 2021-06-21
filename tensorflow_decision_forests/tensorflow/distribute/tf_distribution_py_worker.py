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

"""."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import logging
import tensorflow as tf

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
tf.load_op_library(resource_loader.get_path_to_datafile("distribute.so"))


def main(argv):
  del argv

  cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
  logging.info("Configuration: %s", cluster_resolver.cluster_spec())
  server = tf.distribute.Server(
      cluster_resolver.cluster_spec(),
      job_name=cluster_resolver.task_type,
      task_index=cluster_resolver.task_id,
      protocol=cluster_resolver.rpc_layer or
      "grpc",  # cluster_resolver.rpc_layer or "grpc"
      start=True)
  logging.info("Server started, waiting for jobs")
  server.join()
  logging.info("Shutting down server")


if __name__ == "__main__":
  app.run(main)
