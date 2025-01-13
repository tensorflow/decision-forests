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

"""Blob Sequence reader and writer.

A blob sequence is a stream (e.g. a file) containing a sequence of blob (i.e.
chunk of bytes). It can be used to store sequence of serialized protos.

See yggdrasil_decision_forests/utils/blob_sequence.h for more details.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Iterator, Optional

import tensorflow as tf


class Reader(object):
  """Reader of Blob Sequence files.

  Usage example:

    for blob in Reader(path):
      print(blob)
  """

  def __init__(self, path: str):
    self.file_: tf.io.gfile.Gfile = None
    self.path_ = None

    if path:
      self.open(path)

  def open(self, path: str):
    """Open Blob sequence file."""

    self.file_ = tf.io.gfile.GFile(path, "rb")
    self.path_ = path

    # Reader header.
    magic = self.file_.read(2)
    if magic != b"BS":
      raise ValueError(f"Invalid blob sequence file {path}")
    version = int.from_bytes(self.file_.read(2), byteorder="little")
    if version == 0:
      reserved = self.file_.read(4)
    elif version == 1:
      compression = int.from_bytes(self.file_.read(1), byteorder="little")
      if compression != 0:
        return ValueError(
            "The TF-DF inspector does not support this format of model"
            " (blob-sequence-v1 with compression). Use the format"
            " blob-sequence-v1 without compression instead."
        )
      reserved = self.file_.read(3)
    else:
      raise ValueError(
          f"Non supported blob sequence version {version} for file {path}. The"
          " model was created with a more recent vesion of YDF / TF-DF."
      )
    del reserved

  def close(self):
    self.file_.close()
    self.path_ = None
    self.file_ = None

  def read(self) -> Optional[bytes]:
    """Reads and returns the next blob."""

    raw_length = self.file_.read(4)
    if not raw_length:
      return None
    if len(raw_length) != 4:
      raise ValueError(f"Corrupted blob sequence {self.path_}")
    length = int.from_bytes(raw_length, byteorder="little")
    blob = self.file_.read(length)
    if len(blob) != length:
      raise ValueError(f"Truncated blob sequence {self.path_}")
    return blob

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.file_.close()

  def __iter__(self) -> Iterator[bytes]:
    """Iterates overt the BS file content."""

    # Read blobs
    while True:
      blob = self.read()
      if blob is None:
        break
      yield blob


class Writer(object):
  """Writer of Blob Sequence files.

  Usage example:

    bs = Writer(path)
    bs.write(b"Hello")
    bs.write(b"World")
    bs.close()
  """

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.file_.close()

  def __init__(self, path: Optional[str] = None):
    self.file_: tf.io.gfile.Gfile = None
    self.path_ = None

    if path:
      self.open(path)

  def open(self, path: str):
    self.file_ = tf.io.gfile.GFile(path, "wb")
    self.path_ = path

    self.file_.write(b"BS")
    version = 0
    self.file_.write(version.to_bytes(2, byteorder="little"))
    self.file_.write(b"\0\0\0\0")

  def write(self, blob: bytes):
    self.file_.write(len(blob).to_bytes(4, byteorder="little"))
    self.file_.write(blob)

  def close(self):
    self.file_.close()
    self.path_ = None
    self.file_ = None
