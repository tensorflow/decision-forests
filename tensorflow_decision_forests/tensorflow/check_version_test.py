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

import tensorflow as tf

from tensorflow_decision_forests.tensorflow import check_version


class CheckVersionTest(tf.test.TestCase):

  def test_base(self):

    tf_df_version = "1.2.3"  # Does not matter.
    self.assertTrue(
        check_version.check_version(
            tf_df_version, ["2.6.0", "2.6.1"], "2.6.0", external_logic=True))
    self.assertFalse(
        check_version.check_version(
            tf_df_version, ["2.6.0", "2.6.1"],
            "2.8.0-dev20211105",
            external_logic=True))
    self.assertFalse(
        check_version.check_version(
            tf_df_version, ["2.6.0", "2.6.1"], "2.7.0-rc1",
            external_logic=True))


if __name__ == "__main__":
  tf.test.main()
