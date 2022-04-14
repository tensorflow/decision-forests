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
    """Returns an inline string representation of a tree."""

    root_str = repr(self._root) if self._root else "None"
    return f"Tree(root={root_str}, label_classes={self.label_classes})"

  def __str__(self):
    # This method target users that try to debug or interpret trees.
    return self.pretty()

  def pretty(self, max_depth: Optional[int] = 4) -> str:
    """Returns a readable textual representation of the tree.

    Unlike `repr(tree)`, `tree.pretty()` format the representation (line return,
    margin, hide class names) to improve readability.

    This representation can be changed and codes should not try to parse the
    output of `pretty`. To access programmatically the tree structure, use
    `root()`.

    Args:
      max_depth: The maximum depth of the nodes to display. Deeper nodes are
        skipped and replaced by "...". If not specified, prints the entire tree.

    Returns:
      A pretty-string representing the tree.
    """

    content = ""
    if self._root:
      content += self._root.pretty(
          prefix="", is_pos=None, depth=1, max_depth=max_depth)
    else:
      content += "No root\n"
    if self._label_classes is not None:
      content += f"Label classes: {self.label_classes}\n"
    return content
