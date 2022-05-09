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

from contextlib import contextmanager  # pylint: disable=g-importing-member
import io
import sys
from typing import Union

from tensorflow_decision_forests.tensorflow.ops.training import op as training_op

# Background
# ==========
#
# By default, logs of the Yggdrasil C++ training code are shown on the
# "standard C output" and "error" channels (COUT and CERR). When executing
# python code in a script, those channels are displayed alongside Python
# standard output (i.e. the output of python's "print" function). When running
# in a colab or a notebook, the COUT and CERR channels are not printed i.e. they
# are not visible to the user (unless the user looks in the colab logs). In this
# case, one may want the COUT and CERR channels to be "redirected" to the
# python's standard output.
#
# This parameter
# ==============
#
# If this parameter is set to "auto", and if the code is detected as beeing
# executed in a colab or notebook, the COUT and CERR are redirected to the
# python's standart output.
#
# If this parameter is  True, the COUT and CERR are redirected.
# If this parameter is  False, the COUT and CERR are not redirected.
#
# If the detection of the running environement is incorrect, the training logs
# might not appear in colab (false negative) or the script will hang (stuck when
# the redirection is setup; false positive). If you face one of those
# situations, please ping us.
REDIRECT_YGGDRASIL_CPP_OUTPUT_TO_PYTHON_OUTPUT = "auto"
REDIRECT_MESSAGE_WAS_PRINTED = False


def set_training_logs_redirection(value: Union[str, bool]):
  """Controls the redirection of training logs for display.

  The default value ("auto") should be satifying in most cases.
  If the training is stuck before training, call
  `set_training_logs_redirection(false)`.
  If you don't see the training logs (even through the model created with
  `verbose=2`), call `set_training_logs_redirection(true)`.

  Args:
    value: Redirection. Can be False, True or "auto".
  """

  global REDIRECT_YGGDRASIL_CPP_OUTPUT_TO_PYTHON_OUTPUT
  REDIRECT_YGGDRASIL_CPP_OUTPUT_TO_PYTHON_OUTPUT = value


def info(msg, *args) -> None:
  """Print an info message visible to the user.

  To use instead of absl.logging.info (to be visible in colabs).

  Usage example:
    logging_info("Hello %s", "world")

  Args:
    msg: String message with replacement placeholders e.g. %s.
    *args: Placeholder replacement values.
  """

  print(msg % args, flush=True)


def warning(msg, *args) -> None:
  """Print a warning message visible to the user.

  To use instead of absl.logging.info (to be visible in colabs).

  Usage example:
    logging_warning("Hello %s", "world")

  Args:
    msg: String message with replacement placeholders e.g. %s.
    *args: Placeholder replacement values.
  """

  print("Warning:", msg % args, flush=True)


def capture_cpp_log_context(verbose=False):
  """Creates a context to display or hide the c++ code logs to the user.

  Tested with python, ipython, colab and jupyter notebook.
  In the internal build, only impact python's print.

  Args:
    verbose: If true, the training logs are shown in logging and python's print.
      If false, the training logs are not shown in logging nor python's print.

  Returns:
    A context.
  """

  # Does nothing
  @contextmanager
  def no_op_context():
    yield

  # Hide the Yggdrasil training logs.
  @contextmanager
  def hide_cpp_logs():
    # Stop displaying c++ logs.
    set_yggdrasil_logging_level(0)
    try:
      yield
    finally:
      # Re-start displaying c++ logs.
      set_yggdrasil_logging_level(2)

  if not verbose:
    # Make sure the c++ logs are not visible to the user.
    return hide_cpp_logs()

  def is_direct_output(stream):
    """Checks if output stream redirects to the shell/console directly."""

    if stream.isatty():
      return True
    if isinstance(stream, io.TextIOWrapper):
      return is_direct_output(stream.buffer)
    if isinstance(stream, io.BufferedWriter):
      return is_direct_output(stream.raw)
    if isinstance(stream, io.FileIO):
      return stream.fileno() in [1, 2]
    return False

  if ((REDIRECT_YGGDRASIL_CPP_OUTPUT_TO_PYTHON_OUTPUT == "auto" and
       is_direct_output(sys.stdout)) or
      not REDIRECT_YGGDRASIL_CPP_OUTPUT_TO_PYTHON_OUTPUT):
    # The cout and cerr of the c++ library are already visible to the user.
    return no_op_context()

  global REDIRECT_MESSAGE_WAS_PRINTED
  if not REDIRECT_MESSAGE_WAS_PRINTED:
    REDIRECT_MESSAGE_WAS_PRINTED = True
    info("Standard output detected as not visible to the user e.g. running "
         "in a notebook. Creating a training log redirection. If training get "
         "stuck, try calling tfdf.keras.set_training_logs_redirection(False).")

  # pytype: disable=import-error
  # pylint: disable=g-import-not-at-top
  # pylint: disable=g-importing-member
  # pylint: disable=bare-except

  # The cout and cerr of the c++ library are not visible to the user.
  # Redirect them to python's standard output.
  try:
    from colabtools.googlelog import CaptureLog
    return CaptureLog()
  except:
    try:
      from wurlitzer import sys_pipes
      # This can hang if the cout/cerr is visible to the user.
      return sys_pipes()
    except:
      warning("Cannot redirect the training output because neither of "
              "colabtools.googlelog or wurlitzer available. Run `pip install "
              "wurlitzer -U` and try again.")
      return no_op_context()

  # pylint: enable=g-importing-member
  # pytype: enable=import-error
  # pylint: enable=g-import-not-at-top
  # pylint: enable=bare-except


def set_yggdrasil_logging_level(level: int) -> None:
  """Sets the amount of logging in YggdrasilDecision Forests code.

  No-op in the internal build.

  See: yggdrasil_decision_forests::logging::SetLoggingLevel.

  Args:
    level: Logging level.
  """

  training_op.yggdrasil_decision_forests_set_logging_level(level=level)
