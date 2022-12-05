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

from absl import logging
from absl.testing import parameterized
import tensorflow as tf
from tensorflow_decision_forests.component.tuner import tuner as tuner_lib
from yggdrasil_decision_forests.learner import abstract_learner_pb2

from google.protobuf import text_format


class TunerTest(parameterized.TestCase, tf.test.TestCase):

  def test_base(self):
    tuner = tuner_lib.RandomSearch(
        num_trials=20,
        trial_num_threads=2,
        trial_maximum_training_duration_seconds=10)
    tuner.choice("a", [1, 2, 3])
    tuner.choice("b", [1.0, 2.0, 3.0])
    tuner.choice("c", ["x", "y"])

    s = tuner.choice("c", ["v", "w"], merge=True)
    s.choice("d", [1, 2, 3])

    p = tuner.train_config()
    logging.info("Proto:\n%s", p)

    self.assertEqual(
        p,
        text_format.Parse(
            """
learner: "HYPERPARAMETER_OPTIMIZER"
[yggdrasil_decision_forests.model.hyperparameters_optimizer_v2.proto.hyperparameters_optimizer_config] {
  optimizer {
    optimizer_key: "RANDOM"
    [yggdrasil_decision_forests.model.hyperparameters_optimizer_v2.proto.random] {
      num_trials: 20
    }
  }
  base_learner {
    maximum_training_duration_seconds: 10
  }
  base_learner_deployment {
    num_threads: 2
  }
  search_space {
    fields {
      name: "a"
      discrete_candidates {
        possible_values {
          integer: 1
        }
        possible_values {
          integer: 2
        }
        possible_values {
          integer: 3
        }
      }
    }
    fields {
      name: "b"
      discrete_candidates {
        possible_values {
          real: 1.0
        }
        possible_values {
          real: 2.0
        }
        possible_values {
          real: 3.0
        }
      }
    }
    fields {
      name: "c"
      discrete_candidates {
        possible_values {
          categorical: "x"
        }
        possible_values {
          categorical: "y"
        }
        possible_values {
          categorical: "v"
        }
        possible_values {
          categorical: "w"
        }
      }
      children {
        name: "d"
        discrete_candidates {
          possible_values {
            integer: 1
          }
          possible_values {
            integer: 2
          }
          possible_values {
            integer: 3
          }
        }
        parent_discrete_values {
          possible_values {
            categorical: "v"
          }
          possible_values {
            categorical: "w"
          }
        }
      }
    }
  }
}
     """, abstract_learner_pb2.TrainingConfig()))

  def test_errors(self):
    tuner = tuner_lib.RandomSearch(num_trials=20)
    self.assertRaises(ValueError, lambda: tuner.choice("a", []))
    tuner.choice("a", [1, 2])
    self.assertRaises(ValueError, lambda: tuner.choice("a", [3, 4]))
    tuner.choice("a", [3, 4], merge=True)
    self.assertRaises(ValueError, lambda: tuner.choice("a", [5.0, 6.0]))


if __name__ == "__main__":
  tf.test.main()
