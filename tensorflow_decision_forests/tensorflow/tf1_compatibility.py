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

"""Tensorflow 1 compatibility utilities.

Replaces some TF2 libraries with their TF2 counterpart where necessary.
"""

import tensorflow as tf

if hasattr(tf, '__internal__'):
  Trackable = tf.__internal__.tracking.Trackable
  AutoTrackable = tf.__internal__.tracking.AutoTrackable
  TrackableResource = tf.saved_model.experimental.TrackableResource
  no_automatic_dependency_tracking = tf.__internal__.tracking.no_automatic_dependency_tracking
else:
  # pylint: disable=g-direct-tensorflow-import, disable=g-import-not-at-top
  from tensorflow.python.trackable import autotrackable
  from tensorflow.python.trackable import base as trackable_base
  from tensorflow.python.trackable import resource
  from tensorflow.python.trackable import base as base_tracking
  # pylint: enable=g-direct-tensorflow-import, g-import-not-at-top
  
  Trackable = trackable_base.Trackable
  AutoTrackable = autotrackable.AutoTrackable
  TrackableResource = resource.TrackableResource
  no_automatic_dependency_tracking = base_tracking.no_automatic_dependency_tracking
