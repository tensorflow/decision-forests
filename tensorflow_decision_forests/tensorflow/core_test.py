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
import pandas as pd
import tensorflow as tf

from tensorflow_decision_forests.tensorflow import core
from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.model import hyperparameter_pb2


class SimpleMlTfTest(tf.test.TestCase):

  def test_combine_tensors_and_semantics(self):
    semantic_tensors = core.combine_tensors_and_semantics(
        {
            "a": tf.zeros(shape=[1], dtype=tf.float32),
            "b": tf.zeros(shape=[2], dtype=tf.float32)
        }, {
            "a": core.Semantic.NUMERICAL,
            "b": core.Semantic.CATEGORICAL
        })

    self.assertEqual(semantic_tensors.keys(), {"a", "b"})
    self.assertEqual(semantic_tensors["a"].semantic, core.Semantic.NUMERICAL)
    self.assertEqual(semantic_tensors["a"].tensor.shape, [1])
    self.assertEqual(semantic_tensors["a"].tensor.dtype, tf.float32)

    self.assertEqual(semantic_tensors["b"].semantic, core.Semantic.CATEGORICAL)
    self.assertEqual(semantic_tensors["b"].tensor.shape, [2])
    self.assertEqual(semantic_tensors["b"].tensor.dtype, tf.float32)

  def test_combine_tensors_and_semantics_no_subset(self):
    with self.assertRaises(ValueError):
      core.combine_tensors_and_semantics(
          {
              "a": tf.zeros(shape=[5], dtype=tf.float32),
          }, {"b": core.Semantic.CATEGORICAL})

  def test_decombine_tensors_and_semantics(self):
    tensors, semantics = core.decombine_tensors_and_semantics({
        "a":
            core.SemanticTensor(core.Semantic.NUMERICAL,
                                tf.zeros(shape=[1], dtype=tf.float32)),
        "b":
            core.SemanticTensor(core.Semantic.CATEGORICAL,
                                tf.zeros(shape=[2], dtype=tf.float32))
    })

    self.assertEqual(tensors.keys(), {"a", "b"})
    self.assertEqual(tensors["a"].shape, [1])
    self.assertEqual(tensors["a"].dtype, tf.float32)

    self.assertEqual(tensors["b"].shape, [2])
    self.assertEqual(tensors["b"].dtype, tf.float32)

    self.assertEqual(semantics, {
        "a": core.Semantic.NUMERICAL,
        "b": core.Semantic.CATEGORICAL
    })

  def test_normalize_inputs(self):
    normalized = core.normalize_inputs({
        "a":
            core.SemanticTensor(
                semantic=core.Semantic.NUMERICAL,
                tensor=tf.zeros(shape=[5], dtype=tf.float32)),
        "b":
            core.SemanticTensor(
                semantic=core.Semantic.NUMERICAL,
                tensor=tf.zeros(shape=[5], dtype=tf.int32)),
        "c":
            core.SemanticTensor(
                semantic=core.Semantic.CATEGORICAL,
                tensor=tf.zeros(shape=[5], dtype=tf.int64)),
        "d":
            core.SemanticTensor(
                semantic=core.Semantic.CATEGORICAL,
                tensor=tf.zeros(shape=[5], dtype=tf.string)),
        "e":
            core.SemanticTensor(
                semantic=core.Semantic.NUMERICAL,
                tensor=tf.zeros(shape=[5, 2], dtype=tf.float32)),
    })
    self.assertLen(normalized, 6)

    a = normalized["a"]
    self.assertEqual(a.semantic, core.Semantic.NUMERICAL)
    self.assertEqual(a.tensor.shape, [5])
    self.assertEqual(a.tensor.dtype, tf.float32)

    b = normalized["b"]
    self.assertEqual(b.semantic, core.Semantic.NUMERICAL)
    self.assertEqual(b.tensor.shape, [5])
    self.assertEqual(b.tensor.dtype, tf.float32)

    c = normalized["c"]
    self.assertEqual(c.semantic, core.Semantic.CATEGORICAL)
    self.assertEqual(c.tensor.shape, [5])
    self.assertEqual(c.tensor.dtype, tf.int32)

    d = normalized["d"]
    self.assertEqual(d.semantic, core.Semantic.CATEGORICAL)
    self.assertEqual(d.tensor.shape, [5])
    self.assertEqual(d.tensor.dtype, tf.string)

    e_0 = normalized["e.0"]
    self.assertEqual(e_0.semantic, core.Semantic.NUMERICAL)
    self.assertEqual(e_0.tensor.shape, [5])
    self.assertEqual(e_0.tensor.dtype, tf.float32)

    e_1 = normalized["e.1"]
    self.assertEqual(e_1.semantic, core.Semantic.NUMERICAL)
    self.assertEqual(e_1.tensor.shape, [5])
    self.assertEqual(e_1.tensor.dtype, tf.float32)

  def test_normalize_inputs_regexp(self):
    self.assertEqual(core.normalize_inputs_regexp("e"), r"^e(\.[0-9]+)?$")

  def test_infer_semantic(self):
    semantics = core.infer_semantic({
        "a": tf.zeros(shape=[5], dtype=tf.float32),
        "b": tf.zeros(shape=[5], dtype=tf.int32),
        "c": tf.zeros(shape=[5], dtype=tf.int64),
        "d": tf.zeros(shape=[5], dtype=tf.string),
    })

    self.assertEqual(
        semantics, {
            "a": core.Semantic.NUMERICAL,
            "b": core.Semantic.NUMERICAL,
            "c": core.Semantic.NUMERICAL,
            "d": core.Semantic.CATEGORICAL,
        })

  def infer_semantic_from_dataframe(self):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    semantics = core.infer_semantic_from_dataframe(df)
    self.assertEqual(semantics, {
        "a": core.Semantic.NUMERICAL,
        "b": core.Semantic.CATEGORICAL,
    })

  def test_infer_semantic_manual(self):
    semantics = core.infer_semantic(
        {
            "a": tf.zeros(shape=[5], dtype=tf.int32),
            "b": tf.zeros(shape=[5], dtype=tf.int32),
        },
        manual_semantics={"b": core.Semantic.CATEGORICAL})

    self.assertEqual(semantics, {
        "a": core.Semantic.NUMERICAL,
        "b": core.Semantic.CATEGORICAL,
    })

  def test_infer_semantic_exclude(self):
    semantics = core.infer_semantic(
        {
            "a": tf.zeros(shape=[5], dtype=tf.int32),
            "b": tf.zeros(shape=[5], dtype=tf.int32),
            "c": tf.zeros(shape=[5], dtype=tf.int32),
        },
        manual_semantics={
            "a": None,
            "b": core.Semantic.CATEGORICAL
        },
        exclude_non_specified=True)

    self.assertEqual(semantics, {
        "a": core.Semantic.NUMERICAL,
        "b": core.Semantic.CATEGORICAL,
    })

  def test_infer_semantic_non_existing(self):
    with self.assertRaises(ValueError):
      core.infer_semantic({
          "a": tf.zeros(shape=[5], dtype=tf.int32),
      },
                          manual_semantics={"b": core.Semantic.CATEGORICAL})

  def test_hparams_dict_to_generic_proto(self):
    generic = core.hparams_dict_to_generic_proto({"a": 1.0, "b": 2, "c": "3"})

    expected = hyperparameter_pb2.GenericHyperParameters()
    a = expected.fields.add()
    a.name = "a"
    a.value.real = 1.0

    b = expected.fields.add()
    b.name = "b"
    b.value.integer = 2

    c = expected.fields.add()
    c.name = "c"
    c.value.categorical = "3"

    self.assertEqual(generic, expected)

  def test_column_type_to_semantic(self):
    self.assertEqual(
        core.column_type_to_semantic(data_spec_pb2.ColumnType.NUMERICAL),
        core.Semantic.NUMERICAL)

    self.assertEqual(
        core.column_type_to_semantic(data_spec_pb2.ColumnType.CATEGORICAL),
        core.Semantic.CATEGORICAL)

    self.assertEqual(
        core.column_type_to_semantic(data_spec_pb2.ColumnType.CATEGORICAL_SET),
        core.Semantic.CATEGORICAL_SET)


if __name__ == "__main__":
  tf.test.main()
