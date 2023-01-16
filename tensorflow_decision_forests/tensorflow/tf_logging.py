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

"""Logging primitives.

Replacement of absl's logging primitives that are always visible to the user.
"""


from typing import Any, List
from absl import logging


def info(msg: str, *args: List[Any]) -> None:
  """Print an info message visible to the user.

  To use instead of absl.logging.info (to be visible in colabs).

  Usage example:
    logging_info("Hello %s", "world")

  Args:
    msg: String message with replacement placeholders e.g. %s.
    *args: Placeholder replacement values.
  """

  print(msg % args, flush=True)
  logging.info(msg, *args)


def warning(msg: str, *args: List[Any]) -> None:
  """Print a warning message visible to the user.

  To use instead of absl.logging.info (to be visible in colabs).

  Usage example:
    logging_warning("Hello %s", "world")

  Args:
    msg: String message with replacement placeholders e.g. %s.
    *args: Placeholder replacement values.
  """

  print("Warning:", msg % args, flush=True)
  logging.warning(msg, *args)
