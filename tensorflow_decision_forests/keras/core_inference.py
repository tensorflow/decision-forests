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

"""Keras model with only the inference logic."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from functools import partial  # pylint: disable=g-importing-member
import os
import tempfile
from typing import Optional, List, Dict, Any, Union, Text, Literal
import uuid
import zipfile

import tensorflow as tf

from tensorflow.python.distribute import input_lib
from tensorflow_decision_forests.component.inspector import inspector as inspector_lib
from tensorflow_decision_forests.tensorflow import core_inference as tf_core
from tensorflow_decision_forests.tensorflow import tf_logging
from tensorflow_decision_forests.tensorflow.ops.inference import api as tf_op
from yggdrasil_decision_forests.learner import abstract_learner_pb2
from yggdrasil_decision_forests.model import abstract_model_pb2  # pylint: disable=unused-import
from yggdrasil_decision_forests.utils.distribute.implementations.grpc import grpc_pb2  # pylint: disable=unused-import

layers = tf.keras.layers
models = tf.keras.models
optimizers = tf.keras.optimizers
losses = tf.keras.losses
backend = tf.keras.backend

# The length of a model identifier
MODEL_IDENTIFIER_LENGTH = 16

# Task solved by a model (e.g. classification, regression, ranking);
Task = tf_core.Task
TaskType = "abstract_model_pb2.Task"  # pylint: disable=invalid-name

# A tensorflow feature column.
FeatureColumn = Any

# Semantic of a feature.
#
# The semantic of a feature defines its meaning and constraint how the feature
# is consumed by the model. For example, a feature can has a numerical or
# categorical semantic. The semantic is often related but not equivalent to the
# representation (e.g. float, integer, string).
#
# Each semantic support a different type of representations, tensor formats and
# has specific way to represent and handle missing (and possibly
# out-of-vocabulary) values.
#
# See "smltf.Semantic" for a detailed explanation.
FeatureSemantic = tf_core.Semantic

# Feature name placeholder.
_LABEL = "__LABEL"
_RANK_GROUP = "__RANK_GROUP"
_UPLIFT_TREATMENT = "__UPLIFT_TREATMENT"
_WEIGHTS = "__WEIGHTS"

# This is the list of characters that should not be used as feature name as they
# as not supported by SavedModel serving signatures.
_FORBIDDEN_FEATURE_CHARACTERS = " \t?%,"

# Advanced configuration for the underlying learning library.
YggdrasilDeploymentConfig = abstract_learner_pb2.DeploymentConfig
YggdrasilTrainingConfig = abstract_learner_pb2.TrainingConfig


class AdvancedArguments(object):
  """Advanced control of the model that most users won't need to use.

  Attributes:
    infer_prediction_signature: Instantiate the model graph after training. This
      allows the model to be saved without specifying an input signature and
      without calling "predict", "evaluate". Disabling this logic can be useful
      in two situations: (1) When the exported signature is different from the
      one used during training, (2) When using a fixed-shape pre-processing that
      consume 1 dimensional tensors (as keras will automatically expend its
      shape to rank 2). For example, when using tf.Transform.
    yggdrasil_training_config: Yggdrasil Decision Forests training
      configuration. Expose a few extra hyper-parameters.
      yggdrasil_deployment_config: Configuration of the computing resources used
        to train the model e.g. number of threads. Does not impact the model
        quality.
    fail_on_non_keras_compatible_feature_name: If true (default), training will
      fail if one of the feature name is not compatible with part of the Keras
      API. If false, a warning will be generated instead.
    predict_single_probability_for_binary_classification: Only used for binary
      classification. If true (default), the prediction of a binary class model
      is a tensor of shape [None, 1] containing the probability of the positive
      class (value=1). If false, the prediction of a binary class model is a
      tensor of shape [None, num_classes=2] containing the probability of the
      complementary classes.
    metadata_framework: Metadata describing the framework used to train the
      model.
    metadata_owner: Metadata describing who trained the model.
    populate_history_with_yggdrasil_logs: If false (default) and if a validation
      dataset is provided, populate the model's history with the final
      validation evaluation computed by the Keras metric (i.e. one evaluation).
      If true or if no validation dataset is provided, populate the model's
      history with the yggdrasil training logs. The yggdrasil training logs
      contains more metrics, but those might not be comparable with other non
      TF-DF models.
    disable_categorical_integer_offset_correction: Yggdrasil Decision Forests
      reserves the value 0 of categorical integer features to the OOV item, so
      the value 0 cannot be used directly. If the
      `disable_categorical_integer_offset_correction` is true, a +1 offset might
      be applied before calling the inference code. This attribute should be
      disabled when creating manually a model with categorical integer features.
      Ultimately, Yggdrasil Decision Forests will support the value 0 as a
      normal value and this parameter will be removed. If
      `disable_categorical_integer_offset_correction` is false, this +1 offset
      is never applied.
  """

  def __init__(
      self,
      infer_prediction_signature: Optional[bool] = True,
      yggdrasil_training_config: Optional[YggdrasilTrainingConfig] = None,
      yggdrasil_deployment_config: Optional[YggdrasilDeploymentConfig] = None,
      fail_on_non_keras_compatible_feature_name: Optional[bool] = True,
      predict_single_probability_for_binary_classification: Optional[
          bool] = True,
      metadata_framework: Optional[str] = "TF Keras",
      metadata_owner: Optional[str] = None,
      populate_history_with_yggdrasil_logs: bool = False,
      disable_categorical_integer_offset_correction: bool = False):
    self.infer_prediction_signature = infer_prediction_signature
    self.yggdrasil_training_config = yggdrasil_training_config or abstract_learner_pb2.TrainingConfig(
    )
    self.yggdrasil_deployment_config = yggdrasil_deployment_config or abstract_learner_pb2.DeploymentConfig(
    )
    self.fail_on_non_keras_compatible_feature_name = fail_on_non_keras_compatible_feature_name
    self.predict_single_probability_for_binary_classification = predict_single_probability_for_binary_classification
    self.metadata_framework = metadata_framework
    self.metadata_owner = metadata_owner
    self.populate_history_with_yggdrasil_logs = populate_history_with_yggdrasil_logs
    self.disable_categorical_integer_offset_correction = disable_categorical_integer_offset_correction


class InferenceCoreModel(models.Model):
  """Keras Model V2 wrapper around an Yggdrasil Model.

  See "CoreModel" in "core.py" for the definition of the arguments.
  """

  def __init__(
      self,
      task: Optional[TaskType] = Task.CLASSIFICATION,
      ranking_group: Optional[str] = None,
      verbose: int = 1,
      advanced_arguments: Optional[AdvancedArguments] = None,
      name: Optional[str] = None,
      preprocessing: Optional["models.Functional"] = None,
      postprocessing: Optional["models.Functional"] = None,
      uplift_treatment: Optional[str] = None,
      temp_directory: Optional[str] = None,
  ):
    super(InferenceCoreModel, self).__init__(name=name)

    self._task = task
    self._ranking_group = ranking_group
    self._verbose = verbose
    self._preprocessing = preprocessing
    self._postprocessing = postprocessing
    self._uplift_treatment = uplift_treatment
    self._temp_directory = temp_directory

    if advanced_arguments is None:
      self._advanced_arguments = AdvancedArguments()
    else:
      self._advanced_arguments = copy.deepcopy(advanced_arguments)

    # Copy the metadata
    if (not self._advanced_arguments.yggdrasil_training_config.metadata
        .HasField("framework") and self._advanced_arguments.metadata_framework):
      self._advanced_arguments.yggdrasil_training_config.metadata.framework = self._advanced_arguments.metadata_framework

    if (not self._advanced_arguments.yggdrasil_training_config.metadata
        .HasField("owner") and self._advanced_arguments.metadata_owner):
      self._advanced_arguments.yggdrasil_training_config.metadata.owner = self._advanced_arguments.metadata_owner

    if (self._task == Task.RANKING) != (ranking_group is not None):
      raise ValueError(
          "ranking_key is used iif. the task is RANKING or the loss is a "
          "ranking loss")

      # True iif. the model is trained.
    self._is_trained = tf.Variable(False, trainable=False, name="is_trained")

    # Unique ID to identify the model during training.
    self._training_model_id = generate_training_id()

    # The following fields contain the trained model. They are set during the
    # graph construction and training process.

    # The compiled Yggdrasil model.
    self._model: Optional[tf_op.ModelV2] = None

    # Compiled Yggdrasil model specialized for returning the active leaves.
    # This model is initialized at the first call to "call_get_leaves" or
    # "predict_get_leaves".
    self._model_get_leaves: Optional[tf_op.ModelV2] = None

    # Semantic of the input features.
    # Also defines what are the input features of the model.
    self._semantics: Optional[Dict[Text, FeatureSemantic]] = None

    # List of Yggdrasil feature identifiers i.e. feature seen by the Yggdrasil
    # learner. Those are computed after the preprocessing, unfolding and
    # casting.
    self._normalized_input_keys: Optional[List[Text]] = None

    # Textual description of the model.
    self._description: Optional[Text] = None

  @property
  def task(self) -> Optional[TaskType]:
    """Task to solve (e.g. CLASSIFICATION, REGRESSION, RANKING)."""
    return self._task

  def make_inspector(self) -> inspector_lib.AbstractInspector:
    """Creates an inspector to access the internal model structure.

    Usage example:

    ```python
    inspector = model.make_inspector()
    print(inspector.num_trees())
    print(inspector.variable_importances())
    ```

    Returns:
      A model inspector.
    """

    path = self.yggdrasil_model_path_tensor().numpy().decode("utf-8")
    return inspector_lib.make_inspector(
        path, file_prefix=self.yggdrasil_model_prefix())

  @tf.function(input_signature=[])
  def yggdrasil_model_path_tensor(self) -> Optional[tf.Tensor]:
    """Gets the path to yggdrasil model, if available.

    The effective path can be obtained with:

    ```python
    yggdrasil_model_path_tensor().numpy().decode("utf-8")
    ```

    Returns:
      Path to the Yggdrasil model.
    """

    return self._model._compiled_model._model_loader.get_model_path()  # pylint: disable=protected-access

  def yggdrasil_model_prefix(self) -> str:
    """Gets the prefix of the internal yggdrasil model."""

    return self._model._compiled_model._model_loader.get_model_prefix()  # pylint: disable=protected-access

  def make_predict_function(self):
    """Prediction of the model (!= evaluation)."""

    @tf.function(reduce_retracing=True)
    def predict_function_not_trained(iterator):
      """Prediction of a non-trained model. Returns "zeros"."""

      data = next(iterator)
      x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
      batch_size = _batch_size(x)
      return tf.zeros([batch_size, 1])

    @tf.function(reduce_retracing=True)
    def predict_function_trained(iterator, model):
      """Prediction of a trained model.

      The only difference with "super.make_predict_function()" is that
      "self.predict_function" is not set and that the "distribute_strategy"
      is not used.

      Args:
        iterator: Iterator over the dataset.
        model: Model object.

      Returns:
        Model predictions.
      """

      def run_step(data):
        outputs = model.predict_step(data)
        with tf.control_dependencies(_minimum_control_deps(outputs)):
          model._predict_counter.assign_add(1)  # pylint:disable=protected-access
        return outputs

      data = next(iterator)
      return run_step(data)

    if self._is_trained.value():
      return partial(predict_function_trained, model=self)
    else:
      return predict_function_not_trained

  def make_test_function(self):
    """Predictions for evaluation."""

    @tf.function(reduce_retracing=True)
    def test_function_not_trained(iterator):
      """Evaluation of a non-trained model."""

      next(iterator)
      return {}

    @tf.function(reduce_retracing=True)
    def step_function_trained(model, iterator):
      """Evaluation of a trained model.

      The only difference with "super.make_test_function()" is that
      "self.test_function" is not set.

      Args:
        model: Model object.
        iterator: Iterator over dataset.

      Returns:
        Evaluation metrics.
      """

      def run_step(data):
        outputs = model.test_step(data)
        with tf.control_dependencies(_minimum_control_deps(outputs)):
          model._test_counter.assign_add(1)  # pylint:disable=protected-access
        return outputs

      data = next(iterator)
      outputs = model.distribute_strategy.run(run_step, args=(data,))
      outputs = _reduce_per_replica(
          outputs, self.distribute_strategy, reduction="first")
      return outputs

    if self._is_trained.value():
      # Special case if steps_per_execution is one.
      if (self._steps_per_execution is None or
          self._steps_per_execution.numpy().item() == 1):

        def test_function(iterator):
          """Runs a test execution with a single step."""
          return step_function_trained(self, iterator)

        if not self.run_eagerly:
          test_function = tf.function(test_function, reduce_retracing=True)

        if self._cluster_coordinator:
          return lambda it: self._cluster_coordinator.schedule(  # pylint: disable=g-long-lambda
              test_function, args=(it,))
        else:
          return test_function

      # If we're using a coordinator, use the value of self._steps_per_execution
      # at the time the function is called/scheduled, and not when it is
      # actually executed.
      elif self._cluster_coordinator:

        def test_function(iterator, steps_per_execution):
          """Runs a test execution with multiple steps."""
          for _ in tf.range(steps_per_execution):
            outputs = step_function_trained(self, iterator)
          return outputs

        if not self.run_eagerly:
          test_function = tf.function(test_function, reduce_retracing=True)

        return lambda it: self._cluster_coordinator.schedule(  # pylint: disable=g-long-lambda
            test_function,
            args=(it, self._steps_per_execution.value()))
      else:

        def test_function(iterator):
          """Runs a test execution with multiple steps."""
          for _ in tf.range(self._steps_per_execution):
            outputs = step_function_trained(self, iterator)
          return outputs

        if not self.run_eagerly:
          test_function = tf.function(test_function, reduce_retracing=True)
        return test_function

    else:
      return test_function_not_trained

  @tf.function(reduce_retracing=True)
  def _build_normalized_inputs(self, inputs) -> Dict[str, tf_core.AnyTensor]:
    """Computes the normalized input of the model.

    The normalized inputs are inputs compatible with the Yggdrasil model.

    Args:
      inputs: Input tensors.

    Returns:
      Normalized inputs.
    """

    assert self._semantics is not None
    assert self._model is not None

    if self._preprocessing is not None:
      inputs = self._preprocessing(inputs)

    if isinstance(inputs, dict):
      # Native format
      pass
    elif isinstance(inputs, tf.Tensor):
      if len(self._semantics) != 1:
        raise ValueError(
            "Calling model with input shape different from the "
            "input shape provided during training: Feeding a single array "
            f"{inputs} while the model was trained on {self._semantics}.")
      inputs = {next(iter(self._semantics.keys())): inputs}
    elif isinstance(inputs, list) or isinstance(inputs, tuple):
      # Note: The name of a tensor (value.name) can change between the training
      # and the inference.
      inputs = {str(idx): value for idx, value in enumerate(inputs)}
    else:
      raise ValueError(
          f"The inference input tensor is expected to be a tensor, list of "
          f"tensors or a dictionary of tensors. Got {inputs} instead")

    # Normalize the input tensor to match Yggdrasil requirements.
    semantic_inputs = tf_core.combine_tensors_and_semantics(
        inputs, self._semantics)
    normalized_semantic_inputs = tf_core.normalize_inputs(
        semantic_inputs,
        categorical_integer_offset_correction=not self._advanced_arguments
        .disable_categorical_integer_offset_correction)
    normalized_inputs, _ = tf_core.decombine_tensors_and_semantics(
        normalized_semantic_inputs)

    return normalized_inputs

  @tf.function(reduce_retracing=True)
  def call(self, inputs, training=False):
    """Inference of the model.

    This method is used for prediction and evaluation of a trained model.

    Args:
      inputs: Input tensors.
      training: Is the model being trained. Always False.

    Returns:
      Model predictions.
    """

    del training

    if self._semantics is None:
      tf_logging.warning(
          "The model was called directly (i.e. using `model(data)` instead of "
          "using `model.predict(data)`) before being trained. The model will "
          "only return zeros until trained. The output shape might change "
          "after training %s", inputs)
      return tf.zeros([_batch_size(inputs), 1])

    normalized_inputs = self._build_normalized_inputs(inputs)

    # Apply the model.
    predictions = self._model.apply(normalized_inputs)

    if (self._advanced_arguments
        .predict_single_probability_for_binary_classification and
        self._task == Task.CLASSIFICATION and
        predictions.dense_predictions.shape[1] == 2):
      # Yggdrasil returns the probably of both classes in binary classification.
      # Keras expects only the value (logit or probability) of the "positive"
      # class (value=1).
      predictions = predictions.dense_predictions[:, 1:2]
    else:
      predictions = predictions.dense_predictions

    if self._postprocessing is not None:
      predictions = self._postprocessing(predictions)

    return predictions

  @tf.function(reduce_retracing=True)
  def call_get_leaves(self, inputs):
    """Computes the index of the active leaf in each tree.

    The active leaf is the leave that that receive the example during inference.

    The returned value "leaves[i,j]" is the index of the active leave for the
    i-th example and the j-th tree. Leaves are indexed by depth first
    exploration with the negative child visited before the positive one
    (similarly as "iterate_on_nodes()" iteration). Leaf indices are also
    available with LeafNode.leaf_idx.

    Args:
      inputs: Input tensors. Same signature as the model's "call(inputs)".

    Returns:
      Index of the active leaf for each tree in the model.
    """

    if self._semantics is None:
      tf_logging.warning(
          "The model was called directly using `call_get_leaves` before "
          "being trained. This method will "
          "only return zeros until trained. The output shape might change "
          "after training %s", inputs)
      return tf.zeros([_batch_size(inputs), 1])

    self._ensure_model_get_leaves_ready()
    normalized_inputs = self._build_normalized_inputs(inputs)
    return self._model_get_leaves.apply_get_leaves(normalized_inputs)

  def predict_get_leaves(self, x):
    """Gets the index of the active leaf of each tree.

    The active leaf is the leave that that receive the example during inference.

    The returned value "leaves[i,j]" is the index of the active leave for the
    i-th example and the j-th tree. Leaves are indexed by depth first
    exploration with the negative child visited before the positive one
    (similarly as "iterate_on_nodes()" iteration). Leaf indices are also
    available with LeafNode.leaf_idx.

    Args:
      x: Input samples as a tf.data.Dataset.

    Returns:
      Index of the active leaf for each tree in the model.
    """

    self._ensure_model_get_leaves_ready()

    leaves = []

    for row in x:
      if isinstance(row, tuple):
        # Remove the label and weight.
        row = row[0]
      leaves.append(self.call_get_leaves(row))

    return tf.concat(leaves, axis=0).numpy()

  def _ensure_model_get_leaves_ready(self):
    """Ensures that the model that generates the leaves is available."""

    # TODO: Re-use "_model" if it supports the get-leaves inference.

    if self._model_get_leaves is None:
      self._model_get_leaves = tf_op.ModelV2(
          model_path=self.yggdrasil_model_path_tensor().numpy().decode("utf-8"),
          file_prefix=self.yggdrasil_model_prefix(),
          verbose=False,
          output_types=["LEAVES"])

  def compile(self, metrics=None, weighted_metrics=None):
    """Configure the model for training.

    Unlike for most Keras model, calling "compile" is optional before calling
    "fit".

    Args:
      metrics: List of metrics to be evaluated by the model during training and
        testing.
      weighted_metrics: List of metrics to be evaluated and weighted by
        `sample_weight` or `class_weight` during training and testing.

    Raises:
      ValueError: Invalid arguments.
    """

    super(InferenceCoreModel, self).compile(
        metrics=metrics, weighted_metrics=weighted_metrics)

  def summary(self, line_length=None, positions=None, print_fn=None):
    """Shows information about the model."""

    super(InferenceCoreModel, self).summary(
        line_length=line_length, positions=positions, print_fn=print_fn)

    if print_fn is None:
      print_fn = print

    if self._model is not None:
      print_fn(self._description)

  # TODO: Use Trace Protocol For TF DF custom types to avoid
  # clearing the cache.
  def _clear_function_cache(self):
    """Clear the @tf.function cache and force re-tracing."""

    # pylint: disable=protected-access
    # TODO: Use _variable_creation_fn directly.
    if hasattr(self.call, "_stateful_fn"):
      fn = self.call._stateful_fn
    else:
      fn = self.call._variable_creation_fn

    if fn:
      if hasattr(fn._function_cache, "primary"):
        fn._function_cache.primary.clear()
      else:
        fn._function_cache.clear()
    # pylint: enable=protected-access

  def _extract_sample(self, x):
    """Extracts a sample (e.g.

    batch, row) from the training dataset.

    Returns None is the sample cannot be extracted.

    Args:
      x: Training dataset in the same format as "fit".

    Returns:
      A sample.
    """

    if isinstance(x, tf.data.Dataset):
      return x.take(1)

    if isinstance(x, input_lib.DistributedDatasetsFromFunction):
      try:
        dataset = x._dataset_fn(None)  # pylint: disable=protected-access
        # Extract the example here (instead of inside of "predict") to make
        # sure this operation is done on the chief.
        for row in dataset.take(1):
          x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(row)
          return x
      except Exception:  # pylint: disable=broad-except
        pass

    try:
      # Work for numpy array and TensorFlow Tensors.
      return tf.nest.map_structure(lambda v: v[0:1], x)
    except Exception:  # pylint: disable=broad-except
      pass

    try:
      # Works for list of primitives.
      if isinstance(x, list) and isinstance(x[0],
                                            (int, float, str, bytes, bool)):
        return x[0:1]
    except Exception:  # pylint: disable=broad-except
      pass

    tf_logging.warning("Dataset sampling not implemented for %s", x)
    return None

  def _build(self, x):
    """Build the internal graph similarly as "build" for classical Keras models.

    Compared to the classical build, supports features with dtypes != float32.

    Args:
      x: Training dataset in the same format as "fit".
    """

    if self._verbose >= 1:
      tf_logging.info("Compiling model...")

    # Note: Build does not support dtypes other than float32.
    super(InferenceCoreModel, self).build([])

    # Force the creation of the graph.
    # If a sample cannot be extracted, the graph will be built at the first call
    # to "predict" or "evaluate".
    if self._advanced_arguments.infer_prediction_signature:
      sample = self._extract_sample(x)
      if sample is not None:
        self.predict(sample, verbose=0)

    if self._verbose >= 1:
      tf_logging.info("Model compiled.")

  def _set_from_yggdrasil_model(self,
                                inspector: inspector_lib.AbstractInspector,
                                path: str,
                                file_prefix: Optional[str] = None,
                                input_model_signature_fn: Optional[
                                    tf_core.InputModelSignatureFn] = tf_core
                                .build_default_input_model_signature):

    if not self._is_compiled:
      self.compile()

    features = inspector.features()
    semantics = {
        feature.name: tf_core.column_type_to_semantic(feature.type)
        for feature in features
    }

    self._training_model_id = file_prefix
    self._semantics = semantics
    self._normalized_input_keys = sorted(list(semantics.keys()))
    self._is_trained.assign(True)
    self._model = tf_op.ModelV2(
        model_path=path, verbose=False, file_prefix=file_prefix)

    # Instantiate the model's graph
    input_model_signature = input_model_signature_fn(inspector)

    @tf.function
    def f(x):
      return self(x)

    # Force the tracing of the function (i.e. build the tf-graph) according to
    # the input signature. When a model is serialized (model.save(path)), only
    # the traced functions are exported.
    #
    # https://www.tensorflow.org/guide/function
    _ = f.get_concrete_function(input_model_signature)


def _batch_size(inputs: Union[tf.Tensor, Dict[str, tf.Tensor]]) -> tf.Tensor:
  """Gets the batch size of a tensor or dictionary of tensors.

  Assumes that all the tensors have the same batchsize.

  Args:
    inputs: Dict of tensors.

  Returns:
    The batch size.

  Raises:
    ValueError: Invalid arguments.
  """

  if isinstance(inputs, dict):
    for v in inputs.values():
      return tf.shape(v)[0]
    raise ValueError("Empty input")
  else:
    return tf.shape(inputs)[0]


def pd_dataframe_to_tf_dataset(
    dataframe,
    label: Optional[str] = None,
    task: Optional[TaskType] = Task.CLASSIFICATION,
    max_num_classes: Optional[int] = 100,
    in_place: Optional[bool] = False,
    fix_feature_names: Optional[bool] = True,
    weight: Optional[str] = None,
    batch_size: Optional[int] = 1000) -> tf.data.Dataset:
  """Converts a Panda Dataframe into a TF Dataset compatible with Keras.

  Details:
    - Ensures columns have uniform types.
    - If "label" is provided, separate it as a second channel in the tf.Dataset
      (as expected by Keras).
    - If "weight" is provided, separate it as a third channel in the tf.Dataset
      (as expected by Keras).
    - If "task" is provided, ensure the correct dtype of the label. If the task
      is a classification and the label is a string, integerize the labels. In
      this
      case, the label values are extracted from the dataset and ordered
      lexicographically. Warning: This logic won't work as expected if the
      training and testing dataset contain different label values. In such
      case, it is preferable to convert the label to integers beforehand while
      making sure the same encoding is used for all the datasets.
    - Returns "tf.data.from_tensor_slices"

  Args:
    dataframe: Pandas dataframe containing a training or evaluation dataset.
    label: Name of the label column.
    task: Target task of the dataset.
    max_num_classes: Maximum number of classes for a classification task. A high
      number of unique value / classes might indicate that the problem is a
      regression or a ranking instead of a classification. Set to None to
      disable checking the number of classes.
    in_place: If false (default), the input `dataframe` will not be modified by
      `pd_dataframe_to_tf_dataset`. However, a copy of the dataset memory will
      be made. If true, the dataframe will be modified in-place.
    fix_feature_names: Some feature names are not supported by the SavedModel
      signature. If `fix_feature_names=True` (default) the feature will be
      renamed and made compatible. If `fix_feature_names=False`, the feature
      name will not be changed, but exporting the model might fail (i.e.
      `model.save(...)`).
    weight: Optional name of a column in `dataframe` to use to weight the
      training.
    batch_size: Number of examples in each batch. The size of the batches has no
      impact on the TF-DF training algorithms. However, a small batch size can
      lead to a large overhead when loading the dataset. Defaults to 1000, but
      if `batch_size` is set to `None`, no batching is applied. Note: TF-DF
      expects for the dataset to be batched.

  Returns:
    A TensorFlow Dataset.
  """

  if not in_place:
    dataframe = dataframe.copy(deep=True)

  if label is not None:

    if label not in dataframe.columns:
      raise ValueError(
          f"The label \"{label}\" is not a column of the dataframe.")

    if task == Task.CLASSIFICATION:

      classification_classes = list(dataframe[label].unique())
      if len(classification_classes) > max_num_classes:
        raise ValueError(
            f"The number of unique classes ({len(classification_classes)}) "
            f"exceeds max_num_classes ({max_num_classes}). A high number of "
            "unique value / classes might indicate that the problem is a "
            "regression or a ranking instead of a classification. If this "
            "problem is effectively a classification problem, increase "
            "`max_num_classes`.")

      if dataframe[label].dtypes in [str, object]:
        classification_classes.sort()
        dataframe[label] = dataframe[label].map(classification_classes.index)

      elif dataframe[label].dtypes in [int, float]:
        if (dataframe[label] < 0).any():
          raise ValueError(
              "Negative integer classification label found. Make sure "
              "you label values are positive or stored as string.")

  if weight is not None:
    if weight not in dataframe.columns:
      raise ValueError(
          f"The weight \"{weight}\" is not a column of the dataframe.")

  if fix_feature_names:
    # Rename the features so they are compatible with SaveModel serving
    # signatures.
    rename_mapping = {}
    new_names = set()
    change_any_feature_name = False
    for column in dataframe:
      new_name = column
      for forbidden_character in _FORBIDDEN_FEATURE_CHARACTERS:
        if forbidden_character in new_name:
          change_any_feature_name = True
          new_name = new_name.replace(forbidden_character, "_")
      # Add a tailing "_" until there are not feature name collisions.
      while new_name in new_names:
        new_name += "_"
        change_any_feature_name = True

      rename_mapping[column] = new_name
      new_names.add(new_name)

    dataframe = dataframe.rename(columns=rename_mapping)
    if change_any_feature_name:
      tf_logging.warning(
          "Some of the feature names have been changed automatically to be "
          "compatible with SavedModels because fix_feature_names=True.")

  # Make sure that missing values for string columns are not represented as
  # float(NaN).
  for col in dataframe.columns:
    if dataframe[col].dtype in [str, object]:
      dataframe[col] = dataframe[col].fillna("")

  if label is not None:
    features_dataframe = dataframe.drop(label, axis=1)

    if weight is not None:
      features_dataframe = features_dataframe.drop(weight, axis=1)
      output = (dict(features_dataframe), dataframe[label].values,
                dataframe[weight].values)
    else:
      output = (dict(features_dataframe), dataframe[label].values)

    tf_dataset = tf.data.Dataset.from_tensor_slices(output)

  else:
    if weight is not None:
      raise ValueError(
          "\"weight\" is only supported if the \"label\" is also provided")
    tf_dataset = tf.data.Dataset.from_tensor_slices(dict(dataframe))

  # The batch size does not impact the training of TF-DF.
  if batch_size is not None:
    tf_dataset = tf_dataset.batch(batch_size)

  # Seems to provide a small (measured as ~4% on a 32k rows dataset) speed-up.
  tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

  setattr(tf_dataset, "_tfdf_task", task)
  return tf_dataset


def yggdrasil_model_to_keras_model(
    src_path: str,
    dst_path: str,
    input_model_signature_fn: Optional[tf_core.InputModelSignatureFn] = tf_core
    .build_default_input_model_signature,
    file_prefix: Optional[str] = None,
    verbose: int = 1,
    disable_categorical_integer_offset_correction: bool = False) -> None:
  """Converts an Yggdrasil model into a TensorFlow SavedModel / Keras model.

  Args:
    src_path: Path to input Yggdrasil Decision Forests model. The model can be a
      directory or a zipped file.
    dst_path: Path to output TensorFlow Decision Forests SavedModel model.
    input_model_signature_fn: A lambda that returns the
      (Dense,Sparse,Ragged)TensorSpec (or structure of TensorSpec e.g.
      dictionary, list) corresponding to input signature of the model. If not
      specified, the input model signature is created by
      "build_default_input_model_signature". For example, specify
      "input_model_signature_fn" if an numerical input feature (which is
      consumed as DenseTensorSpec(float32) by default) will be feed differently
      (e.g. RaggedTensor(int64)).
    file_prefix: Prefix of the model files. Auto-detected if None.
    verbose: Verbosity mode. 0 = silent, 1 = small details, 2 = full details.
    disable_categorical_integer_offset_correction: Force the disabling of the
      integer offset correction. See
      disable_categorical_integer_offset_correction in AdvancedArguments for
      more details.
  """

  # Detect the container of the model.
  if os.path.isdir(src_path):
    src_container = "directory"
  elif zipfile.is_zipfile(src_path):
    src_container = "zip"
  else:
    raise ValueError(
        f"The path {src_path} does not look like a yggdrasil-decision-forests "
        "model. An yggdrasil-decision-forests is either a directory or a zip "
        "file containing among other things, a data_spec.pb file.")

  temp_directory = None
  if src_container == "zip":
    # Unzip the model in a temporary directory
    temp_directory = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(src_path, "r") as zip_handle:
      zip_handle.extractall(temp_directory.name)
    src_path = temp_directory.name

  inspector = inspector_lib.make_inspector(src_path, file_prefix=file_prefix)
  objective = inspector.objective()

  model = InferenceCoreModel(
      task=objective.task,
      ranking_group=objective.group
      if objective.task == inspector_lib.Task.RANKING else None,
      verbose=verbose,
      advanced_arguments=AdvancedArguments(
          disable_categorical_integer_offset_correction=disable_categorical_integer_offset_correction
      ))

  model._set_from_yggdrasil_model(  # pylint: disable=protected-access
      inspector,
      src_path,
      file_prefix=file_prefix,
      input_model_signature_fn=input_model_signature_fn)

  model.save(dst_path)
  return


# The following section is a copy of internal Keras functions that are not
# available in the public api.
#
# Keras does not allow projects to depend on the internal api.

# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield


def _minimum_control_deps(outputs):
  """Returns the minimum control dependencies to ensure step succeeded.

  This function is a strict copy of the function of the same name in the keras
  private API:
  third_party/tensorflow/python/keras/engine/training.py
  """

  if tf.executing_eagerly():
    return []  # Control dependencies not needed.
  outputs = tf.nest.flatten(outputs, expand_composites=True)
  for out in outputs:
    # Variables can't be control dependencies.
    if not isinstance(out, tf.Variable):
      return [out]  # Return first Tensor or Op from outputs.
  return []  # No viable Tensor or Op to use for control deps.


def _expand_1d(data):
  """Expands 1-dimensional `Tensor`s into 2-dimensional `Tensor`s.

  This function is a strict copy of the function of the same name in the keras
  private API:
  third_party/tensorflow/python/keras/engine/data_adapter.py
  """

  def _expand_single_1d_tensor(t):
    # Leaves `CompositeTensor`s as-is.
    if (isinstance(t, tf.Tensor) and isinstance(t.shape, tf.TensorShape) and
        t.shape.rank == 1):
      return tf.expand_dims(t, axis=-1)
    return t

  return tf.nest.map_structure(_expand_single_1d_tensor, data)


def _write_scalar_summaries(logs, step):
  for name, value in logs.items():
    if _is_scalar(value):
      tf.scalar("batch_" + name, value, step=step)


def _is_scalar(x):
  return isinstance(x, (tf.Tensor, tf.Variable)) and x.shape.rank == 0


def _is_per_replica_instance(obj):
  return (isinstance(obj, tf.distribute.DistributedValues) and
          isinstance(obj, tf.__internal__.CompositeTensor))


def _reduce_per_replica(values, strategy, reduction="first"):
  """Reduce PerReplica objects.

  Args:
    values: Structure of `PerReplica` objects or `Tensor`s. `Tensor`s are
      returned as-is.
    strategy: `tf.distribute.Strategy` object.
    reduction: One of 'first', 'concat'.

  Returns:
    Structure of `Tensor`s.
  """

  def _reduce(v):
    """Reduce a single `PerReplica` object."""
    if not _is_per_replica_instance(v):
      return v
    elif reduction == "first":
      return strategy.unwrap(v)[0]
    else:
      raise ValueError('`reduction` must be "first" or "concat". Received: '
                       f"reduction={reduction}.")

  return tf.nest.map_structure(_reduce, values)


def generate_training_id() -> str:
  """Generates random hexadecimal string of length `MODEL_IDENTIFIER_LENGTH`."""
  return uuid.uuid4().hex[:MODEL_IDENTIFIER_LENGTH]


# pylint: enable=g-doc-args
# pylint: enable=g-doc-return-or-yield
