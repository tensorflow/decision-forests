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

"""Check that version of TensorFlow is compatible with TF-DF."""

import logging
import tensorflow as tf


def check_version(tf_df_version,
                  compatible_tf_versions,
                  tf_version=None,
                  external_logic=False):
  """Checks the compatibility of the TF version.

  Prints a warning message and return False in care of likely incompatible
  versions.
  """

  if not external_logic:

  if tf_version is None:
    tf_version = tf.__version__
  if tf_version not in compatible_tf_versions:
    logging.warning(
        "TensorFlow Decision Forests %s is compatible with the following "
        "TensorFlow Versions: %s. However, TensorFlow %s was detected. "
        "This can cause issues with the TF API and symbols in the custom C++ "
        "ops. See the TF and TF-DF compatibility table at "
        "https://github.com/tensorflow/decision-forests/blob/main/documentation/known_issues.md#compatibility-table.",
        tf_df_version, compatible_tf_versions, tf_version)
    return False
  return True


def info_fail_to_load_custom_op(exception, filename):
  logging.warning(
      "Failure to load the %s custom c++ tensorflow ops. "
      "This error is likely caused the version of TensorFlow and "
      "TensorFlow Decision Forests are not compatible. Full error:"
      "%s", filename, exception)
