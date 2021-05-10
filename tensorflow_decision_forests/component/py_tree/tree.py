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

"""A decision tree."""

from typing import Optional, List

from tensorflow_decision_forests.component.py_tree import node as node_lib


class Tree(object):
  """A single decision tree."""

  def __init__(self,
               root: Optional[node_lib.AbstractNode],
               label_classes: Optional[List[str]] = None):
    self._root = root
    self._label_classes = label_classes

  @property
  def root(self):
    return self._root

  @property
  def label_classes(self):
    return self._label_classes

  def __repr__(self):
    content = "Tree("
    if self._root:
      content += f"{self._root}"
    else:
      content += "None"

    content += ",label_classes={self.label_classes})"
    return content
