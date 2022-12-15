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

r"""Minimal usage example of Distributed training with TF-DF.

This example trains and exports a Gradient Boosted Tree model.

Usage example:

  For this example, we need a large dataset. If you don't have such dataset
  available, create a synthetic dataset following the instructions in the
  "Synthetic dataset for usage example" below.

  You need to configure TF Parameters servers. See:
  https://www.tensorflow.org/decision_forests/distributed_training
  https://www.tensorflow.org/tutorials/distribute/parameter_server_training

  TF_CONFIG = ...
  # Start the workers
  # ...
  # Run the chief
  python3 distributed_training.py

Synthetic dataset for usage example:

  In this example, we use a synthetic dataset containing 1M examples. This
  dataset is small enought that is could be used without distributed training,
  but this is a good example.

  This dataset is generated with the "synthetic_dataset" tool of YDF.

  Create a file "config.pbtxt" with the content:
    num_examples:1000000
    num_examples_per_shards: 100
    num_numerical:100
    num_categorical:50
    num_categorical_set:0
    num_boolean:50
    categorical_vocab_size:100

  Then run

  bazel run -c opt \
  //external/ydf/yggdrasil_decision_forests/cli/utils:synthetic_dataset -- \
        --alsologtostderr \
        --options=<some path>/config.pbtxt\
        --train=recordio+tfe:<some path>/train@60 \
        --valid=recordio+tfe:<some path>/valid@20 \
        --test=recordio+tfe:<some path>/test@20 \
        --ratio_valid=0.2 \
        --ratio_test=0.2
"""

from absl import app
from absl import logging
import os

import tensorflow as tf
import tensorflow_decision_forests as tfdf


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Configure training
  logging.info("Configure training")
  cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
  strategy = tf.distribute.experimental.ParameterServerStrategy(
      cluster_resolver)
  with strategy.scope():
    model = tfdf.keras.DistributedGradientBoostedTreesModel(
        # Speed-up training by discretizing numerical features.
        force_numerical_discretization=True,
        # Cache directory used to store checkpoints.
        temp_directory="/cns/bh-d/home/gbm/tmp/ttl=15d/tfdf_cache",
        # Number of threads on each worker.
        num_threads=30,
    )
    model.compile(metrics=["accuracy"])

  # Trains the model.

  dataset_path = "/tmp/synthetic_dataset"

  logging.info("Start training")
  model.fit_on_dataset_path(
      train_path=os.path.join(dataset_path, "train@60"),
      valid_path=os.path.join(dataset_path, "valid@20"),
      label_key="LABEL",
      dataset_format="recordio+tfe")

  logging.info("Trained model:")
  model.summary()

  # Access to model metrics.
  inspector = model.make_inspector()
  logging.info("Model self evaluation: %s", inspector.evaluation().to_dict())
  logging.info("Model training logs: %s", inspector.training_logs())
  inspector.export_to_tensorboard(
      "/cns/bh-d/home/gbm/tmp/ttl=15d/tfdf_tensorboard")

  # Exports the model to disk in the SavedModel format for later re-use. This
  # model can be used with TensorFlow Serving and Yggdrasil Decision Forests
  # (https://ydf.readthedocs.io/en/latest/serving_apis.html).
  logging.info("Export model")
  model.save("/cns/bh-d/home/gbm/tmp/ttl=15d/tfdf_model")


if __name__ == "__main__":
  app.run(main)
