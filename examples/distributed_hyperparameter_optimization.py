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

r"""Example of distributed hyper-parameter optimization with TF-DF.

This example trains and exports a Gradient Boosted Tree model.

Usage example:

  You need to configure TF Parameters servers. See:
  https://www.tensorflow.org/decision_forests/distributed_training
  https://www.tensorflow.org/tutorials/distribute/parameter_server_training

  TF_CONFIG = ...
  # Start the workers
  # ...
  # Run the chief
  python3 distributed_hyperparameter_optimization.py
"""

from absl import app
from absl import logging

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
    tuner = tfdf.tuner.RandomSearch(
        # 200 trials to find the best hyper-parameters.
        num_trials=200,
        # Use the pre-defined hyper-parameter space.
        use_predefined_hps=True,
        # Each model is trained on 4 threads.
        trial_num_threads=4)
    model = tfdf.keras.GradientBoostedTreesModel(
        tuner=tuner,
        temp_directory="/cns/bh-d/home/gbm/tmp/ttl=15d/tfdf_cache_dho3",
        # Number of threads available on each worker.
        num_threads=30,
    )

  # Trains the model.
  logging.info("Start tuning")
  model.fit_on_dataset_path(
      train_path="/cns/is-d/home/gbm/ml_dataset_repository/others/adult/adult_train.csv",
      valid_path="/cns/is-d/home/gbm/ml_dataset_repository/others/adult/adult_test.csv",
      label_key="income",
      dataset_format="csv")

  logging.info("Trained model:")
  model.summary()

  # Access to model metrics.
  inspector = model.make_inspector()
  logging.info("Model self evaluation: %s", inspector.evaluation().to_dict())

  # Exports the model to disk in the SavedModel format for later re-use. This
  # model can be used with TensorFlow Serving and Yggdrasil Decision Forests
  # (https://ydf.readthedocs.io/en/latest/serving_apis.html).
  logging.info("Export model")
  model.save("/cns/bh-d/home/gbm/tmp/ttl=15d/tfdf_model_dho3")


if __name__ == "__main__":
  app.run(main)
