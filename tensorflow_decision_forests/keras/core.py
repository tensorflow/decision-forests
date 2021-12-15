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

"""Core wrapper.

This file contains the Keras model wrapper around an Yggdrasil model/learner.
While it can be used directly, the helper functions in keras.py /
wrapper_pre_generated.py should be preferred as they explicit more directly the
learner specific hyper-parameters.

Usage example:

```python
# Indirect usage
import tensorflow_decision_forests as tfdf

model = tfdf.keras.RandomForestModel()
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(...)
model.fit(train_ds)

# Direct usage
import tensorflow_decision_forests as tfdf

model = tfdf.keras.CoreModel(learner="RANDOM_FOREST")
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(...)
model.fit(train_ds)
```

See "CoreModel" for more details
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from datetime import timedelta
import copy
from functools import partial  # pylint: disable=g-importing-member
import inspect
import os
import tempfile
from typing import Optional, List, Dict, Any, Union, Text, Tuple, NamedTuple, Set, Callable
import uuid

from absl import logging
import tensorflow as tf

from tensorflow.python.training.tracking import base as base_tracking  # pylint: disable=g-direct-tensorflow-import
from tensorflow_decision_forests.component.inspector import inspector as inspector_lib
from tensorflow_decision_forests.tensorflow import core as tf_core
from tensorflow_decision_forests.tensorflow.ops.inference import api as tf_op
from tensorflow_decision_forests.tensorflow.ops.training import op as training_op
from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.learner import abstract_learner_pb2
from yggdrasil_decision_forests.model import abstract_model_pb2  # pylint: disable=unused-import
from yggdrasil_decision_forests.utils.distribute.implementations.grpc import grpc_pb2  # pylint: disable=unused-import

layers = tf.keras.layers
models = tf.keras.models
optimizers = tf.keras.optimizers
losses = tf.keras.losses
backend = tf.keras.backend

# Task solved by a model (e.g. classification, regression, ranking);
Task = tf_core.Task
TaskType = "abstract_model_pb2.Task"  # pylint: disable=invalid-name

# Hyper-parameters of the model represented as a dictionary of <parameter names,
# parameter values>.
HyperParameters = tf_core.HyperParameters

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

# Get the current worker index and total number of workers.
get_worker_idx_and_num_workers = tf_core.get_worker_idx_and_num_workers

# check_dataset=True in model constructors enable checks on the dataset
# For example, the dataset should not contain repeat operations and the batch
# size should be large enougth (depending on the number of examples). See the
# "check_dataset" argument documentation for the exact definition.
#
# If the check fails, and if ONLY_WARN_ON_DATASET_CONFIGURATION_ISSUES=If true,
# a warning is printed. Instead, if
# ONLY_WARN_ON_DATASET_CONFIGURATION_ISSUES=false, a ValueException is raised.
#
# TODO(b/206981020): Set False on 18 Jan 2022.
ONLY_WARN_ON_DATASET_CONFIGURATION_ISSUES = True


class FeatureUsage(object):
  """Semantic and hyper-parameters for a single feature.

  This class allows to:
    1. Limit the input features of the model.
    2. Set manually the semantic of a feature.
    3. Specify feature specific hyper-parameters.

  Note that the model's "features" argument is optional. If it is not specified,
  all available feature will be used. See the "CoreModel" class
  documentation for more details.

  Usage example:

  ```python
  # A feature named "A". The semantic will be detected automatically. The
  # global hyper-parameters of the model will be used.
  feature_a = FeatureUsage(name="A")

  # A feature named "C" representing a CATEGORICAL value.
  # Specifying the semantic ensure the feature is correctly detected.
  # In this case, the feature might be stored as an integer, and would have be
  # detected as NUMERICAL.
  feature_b = FeatureUsage(name="B", semantic=Semantic.CATEGORICAL)

  # A feature with a specific maximum dictionary size.
  feature_c = FeatureUsage(name="C",
                                semantic=Semantic.CATEGORICAL,
                                max_vocab_count=32)

  model = CoreModel(features=[feature_a, feature_b, feature_c])
  ```

  Attributes:
    name: The name of the feature. Used as an identifier if the dataset is a
      dictionary of tensors.
    semantic: Semantic of the feature. If None, the semantic is automatically
      determined. The semantic controls how a feature is interpreted by a model.
      Using the wrong semantic (e.g. numerical instead of categorical) will hurt
      your model. See "FeatureSemantic" and "Semantic" for the definition of the
      of available semantics.
    discretized: For NUMERICAL features only. If set, the numerical values are
      discretized into a small set of unique values. This makes the training
      faster but often lead to worst models. A reasonable discretization value
      is 255.
    max_vocab_count: For CATEGORICAL and CATEGORICAL_SET features only. Number
      of unique categorical values stored as string. If more categorical values
      are present, the least frequent values are grouped into a
      Out-of-vocabulary item. Reducing the value can improve or hurt the model.
  """

  def __init__(self,
               name: Text,
               semantic: Optional[FeatureSemantic] = None,
               discretized: Optional[int] = None,
               max_vocab_count: Optional[int] = None):

    self._name = name
    self._semantic = semantic
    self._guide = data_spec_pb2.ColumnGuide()

    # Check matching between hyper-parameters and semantic.
    if semantic != FeatureSemantic.NUMERICAL:
      if discretized is not None:
        raise ValueError("\"discretized\" only works for NUMERICAL semantic.")

    if semantic not in [
        FeatureSemantic.CATEGORICAL, FeatureSemantic.CATEGORICAL_SET
    ]:
      if max_vocab_count is not None:
        raise ValueError("\"max_vocab_count\" only works for CATEGORICAL "
                         "and CATEGORICAL_SET semantic.")

    if semantic is None:
      # The semantic is automatically determined at training time.
      pass

    elif semantic == FeatureSemantic.NUMERICAL:
      self._guide.type = (
          data_spec_pb2.DISCRETIZED_NUMERICAL
          if discretized else data_spec_pb2.NUMERICAL)

    elif semantic in [
        FeatureSemantic.CATEGORICAL, FeatureSemantic.CATEGORICAL_SET
    ]:
      if semantic == FeatureSemantic.CATEGORICAL:
        self._guide.type = data_spec_pb2.CATEGORICAL
      else:
        self._guide.type = data_spec_pb2.CATEGORICAL_SET

      if max_vocab_count:
        self._guide.categorial.max_vocab_count = max_vocab_count

    else:
      raise ValueError("Non supported semantic {}".format(semantic))

  @property
  def guide(self) -> data_spec_pb2.ColumnGuide:  # pylint: disable=g-missing-from-attributes
    return self._guide

  @property
  def semantic(self) -> FeatureSemantic:
    return self._semantic

  @property
  def name(self) -> Text:
    return self._name


class HyperParameterTemplate(NamedTuple):
  """Named and versionned set of hyper-parameters.

  list of hyper-parameter sets that outperforms the default hyper-parameters
  (either generally or in specific scenarios).
  """

  name: str
  version: int
  parameters: Dict[str, Any]
  description: str


class AdvancedArguments(NamedTuple):
  """Advanced control of the model that most users won't need to use.

  Attributes:
    infer_prediction_signature: Instantiate the model graph after training. This
      allows the model to be saved without specifying an input signature and
      without calling "predict", "evaluate". Disabling this logic can be useful
        in two situations: (1) When the exported signature is different from the
          one used during training, (2) When using a fixed-shape pre-processing
          that consume 1 dimensional tensors (as keras will automatically expend
          its shape to rank 2). For example, when using tf.Transform.
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
  """

  infer_prediction_signature: Optional[bool] = True
  yggdrasil_training_config: Optional[
      YggdrasilTrainingConfig] = abstract_learner_pb2.TrainingConfig()
  yggdrasil_deployment_config: Optional[
      YggdrasilDeploymentConfig] = abstract_learner_pb2.DeploymentConfig()
  fail_on_non_keras_compatible_feature_name: Optional[bool] = True
  predict_single_probability_for_binary_classification: Optional[bool] = True
  metadata_framework: Optional[str] = "TF Keras"
  metadata_owner: Optional[str] = None


class CoreModel(models.Model):
  """Keras Model V2 wrapper around an Yggdrasil Learner and Model.

  Basic usage example:

  ```python
  import tensorflow_decision_forests as tfdf

  # Train a classification model with automatic feature discovery.
  model = tfdf.keras.CoreModel(learner="RANDOM_FOREST")
  train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
  model.fit(train_ds)

  # Evaluate the model on another dataset.
  model.evaluate(test_ds)

  # Show information about the model
  model.summary()

  # Export the model with the TF.SavedModel format.
  model.save("/path/to/my/model")
  ```

  The training logs (e.g. feature statistics, validation loss, remaining
  training time) are exported to LOG(INFO). If you use a colab, make sure to
  display these logs:

    from colabtools import googlelog
    with googlelog.CaptureLog():
      model.fit(...)

  Using this model has some caveats:
    * Decision Forest models are not Neural Networks. Feature preprocessing that
      are beneficial to neural network (normalization, one-hot encoding) can be
      detrimental to decision forests. In most cases, it is best to feed the raw
      features (e.g. both numerical and categorical) without preprocessing to
      the model.
    * During training, the entire dataset is loaded in memory (in an efficient
      representation). In case of large datasets (>100M examples), it is
      recommended to randomly downsample the examples.
    * The model trains for exactly one epoch. The core of the training
      computation is done at the end of the first epoch. The console will show
      training logs (including validations losses and feature statistics).
    * The model cannot make predictions before the training is done. Applying
      the model before training will raise an error. During training Keras
      evaluation will be invalid (the model always returns zero).
    * Yggdrasil is itself a C++ model wrapper. Learners and models need to be
      added as dependency to the calling code. To make things practical, the
      Random Forest (without Borg distribution) and Gradient Boosted Decision
      Forest learners and models are linked by default. Other model/learners
      (including yours :)), needs to be added as a dependency manually.

  Attributes:
    task: Task to solve (e.g. CLASSIFICATION, REGRESSION, RANKING).
    learner: The learning algorithm used to train the model. Possible values
      include (but at not limited to) "LEARNER_*".
    learner_params: Hyper-parameters for the learner. The list of available
      hyper-parameters is available at: go/simple_ml/hyper_parameters.
    features: Specify the list and semantic of the input features of the model.
      If not specified, all the available features will be used. If specified
      and if "exclude_non_specified_features=True", only the features in
      "features" will be used by the model. If "preprocessing" is used,
      "features" corresponds to the output of the preprocessing. In this case,
      it is recommended for the preprocessing to return a dictionary of tensors.
    exclude_non_specified_features: If true, only use the features specified in
      "features".
    preprocessing: Functional keras model or @tf.function to apply on the input
      feature before the model to train. This preprocessing model can consume
      and return tensors, list of tensors or dictionary of tensors. If
      specified, the model only "sees" the output of the preprocessing (and not
      the raw input). Can be used to prepare the features or to stack multiple
      models on top of each other. Unlike preprocessing done in the tf.dataset,
      the operation in "preprocessing" are serialized with the model.
    postprocessing: Like "preprocessing" but applied on the model output.
    ranking_group: Only for task=Task.RANKING. Name of a tf.string feature that
      identifies queries in a query/document ranking task. The ranking group is
      not added automatically for the set of features if
      exclude_non_specified_features=false.
    uplift_treatment: Only for task=Task.CATEGORICAL_UPLIFT. Name of an integer
      feature that identifies the treatment in an uplift problem. The value 0 is
      reserved for the control treatment.
    temp_directory: Temporary directory used to store the model Assets after the
      training, and possibly as a work directory during the training. This
      temporary directory is necessary for the model to be exported after
      training e.g. `model.save(path)`. If not specified, `temp_directory` is
      set to a temporary directory using `tempfile.TemporaryDirectory`. This
      directory is deleted when the model python object is garbage-collected.
    verbose: If true, displays information about the training.
    advanced_arguments: Advanced control of the model that most users won't need
      to use. See `AdvancedArguments` for details.
    num_threads: Number of threads used to train the model. Different learning
      algorithms use multi-threading differently and with different degree of
      efficiency. If `None`, `num_threads` will be automatically set to the
      number of processors (up to a maximum of 32; or set to 6 if the number of
      processors is not available). Making `num_threads` significantly larger
      than the number of processors can slow-down the training speed. The
      default value logic might change in the future.
    name: The name of the model.
    max_vocab_count: Default maximum size of the vocabulary for CATEGORICAL and
      CATEGORICAL_SET features stored as strings. If more unique values exist,
      only the most frequent values are kept, and the remaining values are
      considered as out-of-vocabulary. The value `max_vocab_count` defined in a
      `FeatureUsage` (if any) takes precedence.
    try_resume_training: If true, the model training resumes from the checkpoint
      stored in the `temp_directory` directory. If `temp_directory` does not
      contain any model checkpoint, the training start from the beginning.
      Resuming training is useful in the following situations: (1) The training
        was interrupted by the user (e.g. ctrl+c or "stop" button in a
        notebook). (2) the training job was interrupted (e.g. rescheduling), ond
        (3) the hyper-parameter of the model were changed such that an initially
        completed training is now incomplete (e.g. increasing the number of
        trees).
      Note: Training can only be resumed if the training datasets is exactly the
        same (i.e. no reshuffle in the tf.data.Dataset).
    check_dataset: If set to true, test if the dataset is well configured for
      the training: (1) Check if the dataset does contains any `repeat`
        operations, (2) Check if the dataset does contain a `batch` operation,
        (3) Check if the dataset has a large enough batch size (min 100 if the
        dataset contains more than 1k examples or if the number of examples is
        not available) If set to false, do not run any test.
  """

  def __init__(self,
               task: Optional[TaskType] = Task.CLASSIFICATION,
               learner: Optional[str] = "RANDOM_FOREST",
               learner_params: Optional[HyperParameters] = None,
               features: Optional[List[FeatureUsage]] = None,
               exclude_non_specified_features: Optional[bool] = False,
               preprocessing: Optional["models.Functional"] = None,
               postprocessing: Optional["models.Functional"] = None,
               ranking_group: Optional[str] = None,
               uplift_treatment: Optional[str] = None,
               temp_directory: Optional[str] = None,
               verbose: Optional[bool] = True,
               advanced_arguments: Optional[AdvancedArguments] = None,
               num_threads: Optional[int] = None,
               name: Optional[str] = None,
               max_vocab_count: Optional[int] = 2000,
               try_resume_training: Optional[bool] = True,
               check_dataset: Optional[bool] = True) -> None:
    super(CoreModel, self).__init__(name=name)

    self._task = task
    self._learner = learner
    self._learner_params = learner_params
    self._features = features or []
    self._exclude_non_specified = exclude_non_specified_features
    self._preprocessing = preprocessing
    self._postprocessing = postprocessing
    self._ranking_group = ranking_group
    self._uplift_treatment = uplift_treatment
    self._temp_directory = temp_directory
    self._verbose = verbose
    self._num_threads = num_threads
    self._max_vocab_count = max_vocab_count
    self._try_resume_training = try_resume_training
    self._check_dataset = check_dataset

    # Internal, indicates whether the first evaluation during training,
    # triggered by providing validation data, should trigger the training
    # itself.
    self._train_on_evaluate: bool = False

    # Determine the optimal number of threads.
    if self._num_threads is None:
      self._num_threads = os.cpu_count()
      if self._num_threads is None:
        if self._verbose:
          logging.warning(
              "Cannot determine the number of CPUs. Set num_threads=6")
        self._num_threads = 6
      else:
        if self._verbose:

          if self._num_threads >= 32:
            logging.warning(
                "The `num_threads` constructor argument is not set and the "
                "number of CPU is os.cpu_count()=%d > 32. Setting num_threads "
                "to 32. Set num_threads manually to use more than 32 cpus.",
                self._num_threads)
            self._num_threads = 32
          else:
            logging.info("Set num_threads = os.cpu_count() = %d",
                         self._num_threads)

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

    if not self._features and exclude_non_specified_features:
      raise ValueError(
          "The model does not have any input features: "
          "exclude_non_specified_features is True and not features are "
          "provided as input.")

    if self._temp_directory is None:
      self._temp_directory_handle = tempfile.TemporaryDirectory()
      self._temp_directory = self._temp_directory_handle.name
      logging.info("Using %s as temporary training directory",
                   self._temp_directory)

    if (self._task == Task.RANKING) != (ranking_group is not None):
      raise ValueError(
          "ranking_key is used iif. the task is RANKING or the loss is a "
          "ranking loss")

    # True iif. the model is trained.
    self._is_trained = tf.Variable(False, trainable=False, name="is_trained")

    # Unique ID to identify the model during training.
    self._training_model_id = str(uuid.uuid4())

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

    # If the model is trained with weights.
    self._weighted_training = False

    self._time_begin_data_feed: Optional[datetime] = None
    self._time_end_data_feed: Optional[datetime] = None
    self._time_begin_training: Optional[datetime] = None
    self._time_end_training: Optional[datetime] = None

  @property
  def learner_params(self) -> Optional[HyperParameters]:
    """Gets the dictionary of hyper-parameters passed in the model constructor.

    Changing this dictionary will impact the training.
    """
    return self._learner_params

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
    return inspector_lib.make_inspector(path)

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

  def make_predict_function(self):
    """Prediction of the model (!= evaluation)."""

    @tf.function(experimental_relax_shapes=True)
    def predict_function_not_trained(iterator):
      """Prediction of a non-trained model. Returns "zeros"."""

      data = next(iterator)
      x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
      batch_size = _batch_size(x)
      return tf.zeros([batch_size, 1])

    @tf.function(experimental_relax_shapes=True)
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

    if self._is_trained:
      return partial(predict_function_trained, model=self)
    else:
      return predict_function_not_trained

  def make_test_function(self):
    """Predictions for evaluation."""

    @tf.function(experimental_relax_shapes=True)
    def test_function_not_trained(iterator):
      """Evaluation of a non-trained model."""

      next(iterator)
      return {}

    @tf.function(experimental_relax_shapes=True)
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

    if self._is_trained:
      # Special case if steps_per_execution is one.
      if (self._steps_per_execution is None or
          self._steps_per_execution.numpy().item() == 1):

        def test_function(iterator):
          """Runs a test execution with a single step."""
          return step_function_trained(self, iterator)

        if not self.run_eagerly:
          test_function = tf.function(
              test_function, experimental_relax_shapes=True)

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
          test_function = tf.function(
              test_function, experimental_relax_shapes=True)

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
          test_function = tf.function(
              test_function, experimental_relax_shapes=True)
        return test_function

    else:
      return test_function_not_trained

  @tf.function(experimental_relax_shapes=True)
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
      assert len(self._semantics) == 1
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
    normalized_semantic_inputs = tf_core.normalize_inputs(semantic_inputs)
    normalized_inputs, _ = tf_core.decombine_tensors_and_semantics(
        normalized_semantic_inputs)

    return normalized_inputs

  @tf.function(experimental_relax_shapes=True)
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
      logging.warning(
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

  @tf.function(experimental_relax_shapes=True)
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
      logging.warning(
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

    # TODO(gbm): Use get_data_handler when Keras makes it public.

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

    # TODO(gbm): Re-use "_model" if it supports the get-leaves inference.

    if self._model_get_leaves is None:
      self._model_get_leaves = tf_op.ModelV2(
          model_path=self.yggdrasil_model_path_tensor().numpy().decode("utf-8"),
          verbose=False,
          output_types=["LEAVES"])

  # This function should not be serialized in the SavedModel.
  @base_tracking.no_automatic_dependency_tracking
  @tf.function(experimental_relax_shapes=True)
  def train_step(self, data):
    """Collects training examples."""

    if isinstance(data, dict):
      raise ValueError("No label received for training. If you used "
                       "`pd_dataframe_to_tf_dataset`, make sure to "
                       f"specify the `label` argument. data={data}")

    if len(data) == 2:
      train_x, train_y = data
      train_weights = None
    elif len(data) == 3:
      train_x, train_y, train_weights = data
    else:
      raise ValueError(f"Unexpected data shape {data}")

    if self._verbose:
      logging.info("Collect training examples.\nFeatures: %s\nLabel: %s",
                   train_x, train_y)

    if isinstance(train_x, dict):
      _check_feature_names(
          train_x.keys(),
          self._advanced_arguments.fail_on_non_keras_compatible_feature_name)

    if self._preprocessing is not None:
      train_x = self._preprocessing(train_x)
      if self._verbose:
        logging.info("Applying preprocessing on inputs. Result: %s", train_x)
      if isinstance(train_x, list) and self._features:
        logging.warn(
            "Using \"features\" with a pre-processing stage returning a list "
            "is not recommended. Use a pre-processing stage that returns a "
            "dictionary instead.")

    if isinstance(train_x, dict):
      # Native format
      pass
    elif isinstance(train_x, tf.Tensor):
      train_x = {train_x.name: train_x}
    elif isinstance(train_x, list) or isinstance(train_x, tuple):
      # Note: The name of a tensor (value.name) can change between the training
      # and the inference.
      train_x = {str(idx): value for idx, value in enumerate(train_x)}
    else:
      raise ValueError(
          f"The training input tensor is expected to be a tensor, list of "
          f"tensors or a dictionary of tensors. Got {train_x} instead")

    # Check the labels
    if not isinstance(train_y, tf.Tensor):
      raise ValueError(
          f"The training label tensor is expected to be a tensor. Got {train_y}"
          " instead.")

    if len(train_y.shape) != 1:
      if self._verbose:
        logging.info("Squeezing labels to [batch_size] from [batch_size, 1].")
      train_y = tf.squeeze(train_y, axis=1)

    if len(train_y.shape) != 1:
      raise ValueError(
          "Labels can either be passed in as [batch_size, 1] or [batch_size]. "
          "Invalid shape %s." % train_y.shape)

    # Check the training
    self._weighted_training = train_weights is not None
    if self._weighted_training:
      if not isinstance(train_weights, tf.Tensor):
        raise ValueError(
            f"The training weights tensor is expected to be a tensor. Got {train_weights}"
            " instead.")

      if len(train_weights.shape) != 1:
        if self._verbose:
          logging.info("Squeezing labels to [batch_size] from [batch_size, 1].")
        train_weights = tf.squeeze(train_weights, axis=1)

      if len(train_weights.shape) != 1:
        raise ValueError(
            "Weights can either be passed in as [batch_size, 1] or [batch_size]. "
            "Invalid shape %s." % train_weights.shape)

    # List the input features and their semantics.
    assert self._semantics is None, "The model is already trained"
    self._semantics = tf_core.infer_semantic(
        train_x, {feature.name: feature.semantic for feature in self._features},
        self._exclude_non_specified)

    # The ranking group and treatment are not part of the features unless
    # specified explicitly.
    if (self._ranking_group is not None and
        self._ranking_group not in self._features and
        self._ranking_group in self._semantics):
      del self._semantics[self._ranking_group]

    if (self._uplift_treatment is not None and
        self._uplift_treatment not in self._features and
        self._uplift_treatment in self._semantics):
      del self._semantics[self._uplift_treatment]

    semantic_inputs = tf_core.combine_tensors_and_semantics(
        train_x, self._semantics)

    normalized_semantic_inputs = tf_core.normalize_inputs(semantic_inputs)

    if self._verbose:
      logging.info("Normalized features: %s", normalized_semantic_inputs)

    self._normalized_input_keys = sorted(
        list(normalized_semantic_inputs.keys()))

    # Add the weights
    if self._weighted_training:
      normalized_semantic_inputs[_WEIGHTS] = tf_core.SemanticTensor(
          tensor=tf.cast(train_weights, tf_core.NormalizedNumericalType),
          semantic=tf_core.Semantic.NUMERICAL)

    # Add the semantic of the label.
    if self._task == Task.CLASSIFICATION:
      normalized_semantic_inputs[_LABEL] = tf_core.SemanticTensor(
          tensor=tf.cast(train_y, tf_core.NormalizedCategoricalIntType) +
          tf_core.CATEGORICAL_INTEGER_OFFSET,
          semantic=tf_core.Semantic.CATEGORICAL)

    elif self._task == Task.REGRESSION:
      normalized_semantic_inputs[_LABEL] = tf_core.SemanticTensor(
          tensor=tf.cast(train_y, tf_core.NormalizedNumericalType),
          semantic=tf_core.Semantic.NUMERICAL)

    elif self._task == Task.RANKING:
      normalized_semantic_inputs[_LABEL] = tf_core.SemanticTensor(
          tensor=tf.cast(train_y, tf_core.NormalizedNumericalType),
          semantic=tf_core.Semantic.NUMERICAL)

      assert self._ranking_group is not None
      if self._ranking_group not in train_x:
        raise Exception(
            "The ranking key feature \"{}\" is not available as an input "
            "feature.".format(self._ranking_group))
      normalized_semantic_inputs[_RANK_GROUP] = tf_core.SemanticTensor(
          tensor=tf.cast(train_x[self._ranking_group],
                         tf_core.NormalizedHashType),
          semantic=tf_core.Semantic.HASH)

    elif self._task == Task.CATEGORICAL_UPLIFT:
      normalized_semantic_inputs[_LABEL] = tf_core.SemanticTensor(
          tensor=tf.cast(train_y, tf_core.NormalizedCategoricalIntType) +
          tf_core.CATEGORICAL_INTEGER_OFFSET,
          semantic=tf_core.Semantic.CATEGORICAL)

      assert self._uplift_treatment is not None
      if self._uplift_treatment not in train_x:
        raise Exception(
            "The uplift treatment key feature \"{}\" is not available as an input "
            "feature.".format(self._uplift_treatment))
      normalized_semantic_inputs[_UPLIFT_TREATMENT] = tf_core.SemanticTensor(
          tensor=tf.cast(train_x[self._uplift_treatment],
                         tf_core.NormalizedCategoricalIntType) +
          tf_core.CATEGORICAL_INTEGER_OFFSET,
          semantic=tf_core.Semantic.CATEGORICAL)

    else:
      raise Exception("Non supported task {}".format(self._task))

    if not self._is_trained:
      # Collects the training examples.

      distribution_config = tf_core.get_distribution_configuration(
          self.distribute_strategy)
      if distribution_config is None:
        # No distribution strategy. Collecting examples in memory.
        tf_core.collect_training_examples(normalized_semantic_inputs,
                                          self._training_model_id)

      else:

        # Each worker collects a part of the dataset.
        if not self.capabilities().support_partial_cache_dataset_format:
          raise ValueError(
              f"The model {type(self)} does not support training with a TF "
              "Distribution strategy (i.e. model.capabilities()."
              "support_partial_cache_dataset_format == False). If the dataset "
              "is small, simply remove "
              "the distribution strategy scope (i.e. `with strategy.scope():` "
              "around the model construction). If the dataset is large, use a "
              "distributed version of the model. For Example, use "
              "DistributedGradientBoostedTreesModel instead of "
              "GradientBoostedTreesModel.")

        tf_core.collect_distributed_training_examples(
            inputs=normalized_semantic_inputs,
            model_id=self._training_model_id,
            dataset_path=self._distributed_partial_dataset_cache_path())

    # Not metrics are returned during the collection of training examples.
    return {}

  def _distributed_partial_dataset_cache_path(self):
    """Directory accessible from all workers containing the partial cache."""

    return os.path.join(self._temp_directory, "partial_dataset_cache")

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

    super(CoreModel, self).compile(
        metrics=metrics, weighted_metrics=weighted_metrics)

  def fit(self,
          x=None,
          y=None,
          callbacks=None,
          **kwargs) -> tf.keras.callbacks.History:
    """Trains the model.

    The following dataset formats are supported:

      1. "x" is a tf.data.Dataset containing a tuple "(features, labels)".
         "features" can be a dictionary a tensor, a list of tensors or a
         dictionary of tensors (recommended). "labels" is a tensor.

      2. "x" is a tensor, list of tensors or dictionary of tensors containing
         the input features. "y" is a tensor.

      3. "x" is a numpy-array, list of numpy-arrays or dictionary of
         numpy-arrays containing the input features. "y" is a numpy-array.

    Unlike classical neural networks, the learning algorithm requires to scan
    the training dataset exactly once. Therefore, the dataset should not be
    repeated. The algorithm also does not benefit from shuffling the dataset.

    Input features generally do not need to be normalized (numerical) or indexed
    (categorical features stored as string). Also, missing values are well
    supported (i.e. not need to replace missing values).

    Pandas Dataframe can be prepared with "dataframe_to_tf_dataset":
      dataset = pandas.Dataframe(...)
      model.fit(pd_dataframe_to_tf_dataset(dataset, label="my_label"))

    Some of the learning algorithm will support distributed training with the
    ParameterServerStrategy e.g.:

      with tf.distribute.experimental.ParameterServerStrategy(...).scope():
        model = DistributedGradientBoostedTreesModel()
      model.fit(...)

    Args:
      x: Training dataset (See details above for the supported formats).
      y: Label of the training dataset. Only used if "x" does not contains the
        labels.
      callbacks: Callbacks triggered during the training.
      **kwargs: Arguments passed to the core keras model's fit.

    Returns:
      A `History` object. Its `History.history` attribute is not yet
      implemented for decision forests algorithms, and will return empty.
      All other fields are filled as usual for `Keras.Mode.fit()`.
    """

    self._clear_function_cache()

    # Check for a Pandas Dataframe without injecting a dependency.
    if str(type(x)) == "<class 'pandas.core.frame.DataFrame'>":
      raise ValueError(
          "`fit` cannot consume Pandas' dataframes directly. Instead, use the "
          "`pd_dataframe_to_tf_dataset` utility function. For example: "
          "`model.fit(tfdf.keras.pd_dataframe_to_tf_dataset(train_dataframe, "
          "label=\"label_column\"))")

    # If the dataset was created with "pd_dataframe_to_tf_dataset", ensure that
    # the task is correctly set.
    if hasattr(x, "_tfdf_task"):
      dataset_task = getattr(x, "_tfdf_task")
      if dataset_task != self._task:
        raise ValueError(
            f"The model's `task` attribute ({Task.Name(self._task)}) does "
            "not match the `task` attribute passed to "
            f"`pd_dataframe_to_tf_dataset` ({Task.Name(dataset_task)}).")

    # Check the dataset.
    if self._check_dataset and isinstance(x, tf.data.Dataset):
      _check_dataset(x)

    # Call "compile" if the user forgot to do so.
    if not self._is_compiled:
      self.compile()

    if "epochs" in kwargs:
      if kwargs["epochs"] != 1:
        raise ValueError("all decision forests algorithms train with only 1 " +
                         "epoch, epochs={} given".format(kwargs["epochs"]))
      del kwargs["epochs"]  # Not needed since we force it to 1 below.

    # This callback will trigger the training at the end of the first epoch.
    callbacks = [_TrainerCallBack(self)] + (callbacks if callbacks else [])

    # We want the model trained before any evaluation is done at the
    # end of the epoch. This may fail in case any of the `on_train_batch_*`
    # callbacks calls `evaluate()` before the end of the 1st epoch.
    self._train_on_evaluate = True

    # Reset the training status.
    self._is_trained.assign(False)

    try:
      history = super(CoreModel, self).fit(
          x=x, y=y, epochs=1, callbacks=callbacks, **kwargs)
    finally:
      self._train_on_evaluate = False

    self._build(x)

    return history

  def fit_on_dataset_path(
      self,
      train_path: str,
      label_key: str,
      weight_key: Optional[str] = None,
      ranking_key: Optional[str] = None,
      valid_path: Optional[str] = None,
      dataset_format: Optional[str] = "csv",
      max_num_scanned_rows_to_accumulate_statistics: Optional[int] = 100_000,
      try_resume_training: Optional[bool] = True,
      input_model_signature_fn: Optional[tf_core.InputModelSignatureFn] = (
          tf_core.build_default_input_model_signature)):
    """Trains the model on a dataset stored on disk.

    This solution is generally more efficient and easier that loading the
    dataset with a tf.Dataset both for local and distributed training.

    Usage example:

      # Local training
      model = model = keras.GradientBoostedTreesModel()
      model.fit_on_dataset_path(
        train_path="/path/to/dataset.csv",
        label_key="label",
        dataset_format="csv")
      model.save("/model/path")

      # Distributed training
      with tf.distribute.experimental.ParameterServerStrategy(...).scope():
        model = model = keras.DistributedGradientBoostedTreesModel()
      model.fit_on_dataset_path(
        train_path="/path/to/dataset@10",
        label_key="label",
        dataset_format="tfrecord+tfe")
      model.save("/model/path")

    Args:
       train_path: Path to the training dataset. Support comma separated files,
         shard and glob notation.
       label_key: Name of the label column.
       weight_key: Name of the weighing column.
       ranking_key: Name of the ranking column.
       valid_path: Path to the validation dataset. If not provided, or if the
         learning algorithm does not support/need a validation dataset,
         `valid_path` is ignored.
       dataset_format: Format of the dataset. Should be one of the registered
         dataset format (see
         https://github.com/google/yggdrasil-decision-forests/blob/main/documentation/user_manual.md#dataset-path-and-format
           for more details). The format "csv" always available but it is
           generally only suited for small datasets.
      max_num_scanned_rows_to_accumulate_statistics: Maximum number of examples
        to scan to determine the statistics of the features (i.e. the dataspec,
        e.g. mean value, dictionaries). (Currently) the "first" examples of the
        dataset are scanned (e.g. the first examples of the dataset is a single
        file). Therefore, it is important that the sampled dataset is relatively
        uniformly sampled, notably the scanned examples should contains all the
        possible categorical values (otherwise the not seen value will be
        treated as out-of-vocabulary). If set to None, the entire dataset is
        scanned. This parameter has no effect if the dataset is stored in a
        format that already contains those values.
      try_resume_training: If true, tries to resume training from the model
        checkpoint stored in the `temp_directory` directory. If `temp_directory`
        does not contain any model checkpoint, start the training from the
        start. Works in the following three situations: (1) The training was
          interrupted by the user (e.g. ctrl+c). (2) the training job was
          interrupted (e.g. rescheduling), ond (3) the hyper-parameter of the
          model were changed such that an initially completed training is now
          incomplete (e.g. increasing the number of trees).
      input_model_signature_fn: A lambda that returns the
        (Dense,Sparse,Ragged)TensorSpec (or structure of TensorSpec e.g.
        dictionary, list) corresponding to input signature of the model. If not
        specified, the input model signature is created by
        "build_default_input_model_signature". For example, specify
        "input_model_signature_fn" if an numerical input feature (which is
        consumed as DenseTensorSpec(float32) by default) will be feed
        differently (e.g. RaggedTensor(int64)).

    Returns:
      A `History` object. Its `History.history` attribute is not yet
      implemented for decision forests algorithms, and will return empty.
      All other fields are filled as usual for `Keras.Mode.fit()`.
    """

    self._time_begin_training = datetime.now()

    if self._verbose:
      logging.info("Training on dataset %s", train_path)

    self._clear_function_cache()

    # Call "compile" if the user forgot to do so.
    if not self._is_compiled:
      self.compile()

    train_model_path = self._temp_directory
    model_path = os.path.join(train_model_path, "model")

    # Create the dataspec guide.
    guide = data_spec_pb2.DataSpecificationGuide(
        ignore_columns_without_guides=self._exclude_non_specified,
        max_num_scanned_rows_to_accumulate_statistics=max_num_scanned_rows_to_accumulate_statistics
    )
    guide.default_column_guide.categorial.max_vocab_count = self._max_vocab_count
    self._normalized_input_keys = []
    for feature in self._features:
      col_guide = copy.deepcopy(feature.guide)
      col_guide.column_name_pattern = tf_core.normalize_inputs_regexp(
          feature.name)
      guide.column_guides.append(col_guide)
      self._normalized_input_keys.append(feature.name)

    label_guide = data_spec_pb2.ColumnGuide(
        column_name_pattern=tf_core.normalize_inputs_regexp(label_key))

    if self._task == Task.CLASSIFICATION:
      label_guide.type = data_spec_pb2.CATEGORICAL
      label_guide.categorial.min_vocab_frequency = 0
      label_guide.categorial.max_vocab_count = -1
    elif self._task == Task.REGRESSION:
      label_guide.type = data_spec_pb2.NUMERICAL
    elif self._task == Task.RANKING:
      label_guide.type = data_spec_pb2.NUMERICAL
    else:
      raise ValueError(
          f"Non implemented task {self._task} with \"fit_on_dataset_path\"."
          " Use a different task or train with \"fit\".")
    guide.column_guides.append(label_guide)

    if ranking_key:
      ranking_guide = data_spec_pb2.ColumnGuide(
          column_name_pattern=tf_core.normalize_inputs_regexp(ranking_key),
          type=data_spec_pb2.HASH)
      guide.column_guides.append(ranking_guide)

    if weight_key:
      weight_guide = data_spec_pb2.ColumnGuide(
          column_name_pattern=tf_core.normalize_inputs_regexp(weight_key),
          type=data_spec_pb2.NUMERICAL)
      guide.column_guides.append(weight_guide)

    # Deployment configuration
    deployment_config = copy.deepcopy(
        self._advanced_arguments.yggdrasil_deployment_config)
    if not deployment_config.HasField("num_threads"):
      deployment_config.num_threads = self._num_threads

    distribution_config = tf_core.get_distribution_configuration(
        self.distribute_strategy)
    logging.info("distribution_config: %s", distribution_config)
    if distribution_config is not None and not self.capabilities(
    ).support_partial_cache_dataset_format:
      raise ValueError(
          f"The model {type(self)} does not support training with a TF "
          "Distribution strategy (i.e. model.capabilities()."
          "support_partial_cache_dataset_format == False). If the dataset "
          "is small, simply remove the distribution strategy scope (i.e. `with "
          "strategy.scope():` around the model construction). If the dataset "
          "is large, use a distributed version of the model. For Example, use "
          "DistributedGradientBoostedTreesModel instead of "
          "GradientBoostedTreesModel.")

    # Train the model.
    tf_core.train_on_file_dataset(
        train_dataset_path=dataset_format + ":" + train_path,
        valid_dataset_path=(dataset_format + ":" +
                            valid_path) if valid_path else None,
        feature_ids=self._normalized_input_keys,
        label_id=label_key,
        weight_id=weight_key,
        model_id=self._training_model_id,
        model_dir=train_model_path,
        learner=self._learner,
        task=self._task,
        generic_hparms=tf_core.hparams_dict_to_generic_proto(
            self._learner_params),
        ranking_group=ranking_key,
        keep_model_in_resource=True,
        guide=guide,
        training_config=self._advanced_arguments.yggdrasil_training_config,
        deployment_config=deployment_config,
        working_cache_path=os.path.join(self._temp_directory, "working_cache"),
        distribution_config=distribution_config,
        try_resume_training=try_resume_training)

    if self._verbose:
      logging.info("Training done. Finalizing the model.")

    # Request and store a description of the model.
    self._description = training_op.SimpleMLShowModel(
        model_identifier=self._training_model_id).numpy().decode("utf-8")
    training_op.SimpleMLUnloadModel(model_identifier=self._training_model_id)

    # Build the model's graph.
    inspector = inspector_lib.make_inspector(model_path)
    self._set_from_yggdrasil_model(
        inspector,
        model_path,
        input_model_signature_fn=input_model_signature_fn)

    # Build the model history.
    history = tf.keras.callbacks.History()
    history.model = self
    history.on_train_begin()

    training_logs = inspector.training_logs()
    if training_logs is not None:
      for src_logs in training_logs:
        if src_logs.evaluation is not None:
          history.on_epoch_end(src_logs.num_trees,
                               src_logs.evaluation.to_dict())
    self.history = history

    self._time_end_training = datetime.now()
    if self._verbose:
      self._print_timer_training()

    return self.history

  def save(self, filepath: str, overwrite: Optional[bool] = True, **kwargs):
    """Saves the model as a TensorFlow SavedModel.

    The exported SavedModel contains a standalone Yggdrasil Decision Forests
    model in the "assets" sub-directory. The Yggdrasil model can be used
    directly using the Yggdrasil API. However, this model does not contain the
    "preprocessing" layer (if any).

    Args:
      filepath: Path to the output model.
      overwrite: If true, override an already existing model. If false, raise an
        error if a model already exist.
      **kwargs: Arguments passed to the core keras model's save.
    """

    # TF does not override assets when exporting a model in a directory already
    # containing a model. In such case, we need to remove the initial assets
    # directory manually.
    # Only the assets directory is removed (instead of the whole "filepath") in
    # case this directory contains important files.
    assets_dir = os.path.join(filepath, "assets")
    saved_model_file = os.path.join(filepath, "saved_model.pb")

    if tf.io.gfile.exists(saved_model_file) and tf.io.gfile.exists(assets_dir):
      if overwrite:
        tf.io.gfile.rmtree(assets_dir)
      else:
        raise ValueError(
            f"A model already exist as {filepath}. Use an empty directory "
            "or set overwrite=True")

    super(CoreModel, self).save(
        filepath=filepath, overwrite=overwrite, **kwargs)

  def evaluate(self, *args, **kwargs):
    """Returns the loss value & metrics values for the model.

    See details on `keras.Model.evaluate`.

    Args:
      *args: Passed to `keras.Model.evaluate`.
      **kwargs: Passed to `keras.Model.evaluate`.  Scalar test loss (if the
        model has a single output and no metrics) or list of scalars (if the
        model has multiple outputs and/or metrics). See details in
        `keras.Model.evaluate`.
    """
    if self._train_on_evaluate:
      if not self._is_trained.numpy():
        self._train_model()
      else:
        raise ValueError(
            "evaluate() requested training of an already trained model -- "
            "did you call `Model.evaluate` from a `on_train_batch*` callback ?"
            "this is not yet supported in Decision Forests models, where one "
            "can only evaluate after the first epoch is finished and the "
            "model trained")
    return super(CoreModel, self).evaluate(*args, **kwargs)

  def summary(self, line_length=None, positions=None, print_fn=None):
    """Shows information about the model."""

    super(CoreModel, self).summary(
        line_length=line_length, positions=positions, print_fn=print_fn)

    if print_fn is None:
      print_fn = print

    if self._model is not None:
      print_fn(self._description)

  @staticmethod
  def predefined_hyperparameters() -> List[HyperParameterTemplate]:
    """Returns a better than default set of hyper-parameters.

    They can be used directly with the `hyperparameter_template` argument of the
    model constructor.

    These hyper-parameters outperforms the default hyper-parameters (either
    generally or in specific scenarios). Like default hyper-parameters, existing
    pre-defined hyper-parameters cannot change.
    """

    return []

  # TODO(b/205971333): Use Trace Protocol For TF DF custom types to avoid
  # clearing the cache.
  def _clear_function_cache(self):
    """Clear the @tf.function cache and force re-tracing."""

    # pylint: disable=protected-access
    if self.call._stateful_fn:
      if hasattr(self.call._stateful_fn._function_cache, "primary"):
        self.call._stateful_fn._function_cache.primary.clear()
      else:
        self.call._stateful_fn._function_cache.clear()
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

    logging.warning("Dataset sampling not implemented for %s", x)
    return None

  def _build(self, x):
    """Build the internal graph similarly as "build" for classical Keras models.

    Compared to the classical build, supports features with dtypes != float32.

    Args:
      x: Training dataset in the same format as "fit".
    """

    # Note: Build does not support dtypes other than float32.
    super(CoreModel, self).build([])

    # Force the creation of the graph.
    # If a sample cannot be extracted, the graph will be built at the first call
    # to "predict" or "evaluate".
    if self._advanced_arguments.infer_prediction_signature:
      sample = self._extract_sample(x)
      if sample is not None:
        self.predict(sample)

  def _train_model(self):
    """Effectively train the model."""

    if self._normalized_input_keys is None:
      raise Exception("The training graph was not built.")

    self._time_end_data_feed = datetime.now()
    if self._verbose:
      self._print_timer_feed_data()
      logging.info("Starting training the model")

    self._time_begin_training = datetime.now()

    train_model_path = self._temp_directory
    model_path = os.path.join(train_model_path, "model")

    # Create the dataspec guide.
    guide = data_spec_pb2.DataSpecificationGuide()
    guide.default_column_guide.categorial.max_vocab_count = self._max_vocab_count
    for feature in self._features:
      col_guide = copy.deepcopy(feature.guide)
      col_guide.column_name_pattern = tf_core.normalize_inputs_regexp(
          feature.name)
      guide.column_guides.append(col_guide)

    # Deployment configuration
    deployment_config = copy.deepcopy(
        self._advanced_arguments.yggdrasil_deployment_config)
    if not deployment_config.HasField("num_threads"):
      deployment_config.num_threads = self._num_threads

    distribution_config = tf_core.get_distribution_configuration(
        self.distribute_strategy)
    if distribution_config is None:
      # Train the model.
      # The model will be exported to "train_model_path".
      #
      # Note: It would be possible to train and load the model without saving
      # the model to file.
      tf_core.train(
          input_ids=self._normalized_input_keys,
          label_id=_LABEL,
          weight_id=_WEIGHTS if self._weighted_training else None,
          model_id=self._training_model_id,
          model_dir=train_model_path,
          learner=self._learner,
          task=self._task,
          generic_hparms=tf_core.hparams_dict_to_generic_proto(
              self._learner_params),
          ranking_group=_RANK_GROUP if self._task == Task.RANKING else None,
          uplift_treatment=_UPLIFT_TREATMENT
          if self._task == Task.CATEGORICAL_UPLIFT else None,
          keep_model_in_resource=True,
          guide=guide,
          training_config=self._advanced_arguments.yggdrasil_training_config,
          deployment_config=deployment_config,
          try_resume_training=self._try_resume_training)

    else:
      tf_core.finalize_distributed_dataset_collection(
          cluster_coordinator=self._cluster_coordinator,
          input_ids=self._normalized_input_keys + [_LABEL] +
          ([_WEIGHTS] if self._weighted_training else []),
          model_id=self._training_model_id,
          dataset_path=self._distributed_partial_dataset_cache_path())

      tf_core.train_on_file_dataset(
          train_dataset_path="partial_dataset_cache:" +
          self._distributed_partial_dataset_cache_path(),
          valid_dataset_path=None,
          feature_ids=self._normalized_input_keys,
          label_id=_LABEL,
          weight_id=_WEIGHTS if self._weighted_training else None,
          model_id=self._training_model_id,
          model_dir=train_model_path,
          learner=self._learner,
          task=self._task,
          generic_hparms=tf_core.hparams_dict_to_generic_proto(
              self._learner_params),
          ranking_group=_RANK_GROUP if self._task == Task.RANKING else None,
          uplift_treatment=_UPLIFT_TREATMENT
          if self._task == Task.CATEGORICAL_UPLIFT else None,
          keep_model_in_resource=True,
          guide=guide,
          training_config=self._advanced_arguments.yggdrasil_training_config,
          deployment_config=deployment_config,
          working_cache_path=os.path.join(self._temp_directory,
                                          "working_cache"),
          distribution_config=distribution_config,
          try_resume_training=self._try_resume_training)

    # Request and store a description of the model.
    self._description = training_op.SimpleMLShowModel(
        model_identifier=self._training_model_id).numpy().decode("utf-8")
    training_op.SimpleMLUnloadModel(model_identifier=self._training_model_id)

    self._is_trained.assign(True)

    self._time_end_training = datetime.now()
    if self._verbose:
      self._print_timer_training()

    # Load and optimize the model in memory.
    # Register the model as a SavedModel asset.
    self._model = tf_op.ModelV2(model_path=model_path, verbose=False)

  def _set_from_yggdrasil_model(self,
                                inspector: inspector_lib.AbstractInspector,
                                path: str,
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

    self._semantics = semantics
    self._normalized_input_keys = sorted(list(semantics.keys()))
    self._is_trained.assign(True)
    self._model = tf_op.ModelV2(model_path=path, verbose=False)

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

  @staticmethod
  def capabilities() -> abstract_learner_pb2.LearnerCapabilities:
    """Lists the capabilities of the learning algorithm."""

    return abstract_learner_pb2.LearnerCapabilities()

  def _print_timer_feed_data(self):

    if self._time_end_data_feed and self._time_begin_data_feed:
      logging.info("Read training dataset in %s",
                   self._time_end_data_feed - self._time_begin_data_feed)

  def _print_timer_training(self):

    if self._time_end_training and self._time_begin_training:
      logging.info("Training model in %s",
                   self._time_end_training - self._time_begin_training)

      # Comparison to data feed stage.
      if self._time_end_data_feed and self._time_begin_data_feed:
        duration_training = self._time_end_training - self._time_begin_training
        duration_data_feed = self._time_end_data_feed - self._time_begin_data_feed
        # Note: The tf tracing takes ~5s with 20 features.
        warning_offset = timedelta(seconds=5.0)
        if duration_training >= timedelta(seconds=0.1):
          ratio = duration_data_feed / (
              duration_data_feed + duration_training + warning_offset)
          if ratio > 0.5:
            logging.warning(
                "Tracing the TF graph and reading the dataset took more than "
                "50%% of the time to effectively train the model "
                "(tracing+dataset reading: %s, training: %s). This might "
                "indicates that the dataset reading operation e.g. "
                "tf.data.Dataset is not well configured. "
                "In mose cases, this ratio should be <<10%%.",
                duration_data_feed, duration_training)


class _TrainerCallBack(tf.keras.callbacks.Callback):
  """Callback that trains the model at the end of the first epoch."""

  def __init__(self, model: CoreModel):
    self._model = model

  def on_epoch_begin(self, epoch, logs=None):
    del logs
    if epoch == 0 and not self._model._is_trained.numpy():
      if self._model._verbose:
        logging.info("Starting reading the dataset")
      self._model._time_begin_data_feed = datetime.now()

  def on_epoch_end(self, epoch, logs=None):
    del logs
    if epoch == 0 and not self._model._is_trained.numpy():  # pylint:disable=protected-access
      self._model._train_model()  # pylint:disable=protected-access

    # After this the model is trained, and evaluations shouldn't attempt
    # to retrain.
    self._model._train_on_evaluate = False  # pylint:disable=protected-access


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
      a classification and the label a string, integerize the labels. In this
      case, the label values are extracted from the dataset and ordered
      lexicographically. Warning: This logic won't work as expected if the
      training and testing dataset contains different label values. In such
      case, it is preferable to convert the label to integers beforehand while
      making sure the same encoding is used for all the datasets. If "
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
      be made. If true, the dataframe will be modified in place.
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

      classification_classes = dataframe[label].unique().tolist()
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
      logging.warning(
          "Some of the feature names have been changed automatically to be "
          "compatible with SavedModels because fix_feature_names=True.")

  # Make sure that missing values for string columns are not represented as
  # float(NaN).
  for col in dataframe.columns:
    if dataframe[col].dtype in [str, object]:
      dataframe[col] = dataframe[col].fillna("")

  if label is not None:
    features_dataframe = dataframe.drop(label, 1)

    if weight is not None:
      features_dataframe = features_dataframe.drop(weight, 1)
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
    .build_default_input_model_signature):
  """Converts an Yggdrasil model into a Keras model.

  Args:
    src_path: Path to input Yggdrasil Decision Forests model.
    dst_path: Path to output TensorFlow Decision Forests SavedModel model.
    input_model_signature_fn: A lambda that returns the
      (Dense,Sparse,Ragged)TensorSpec (or structure of TensorSpec e.g.
      dictionary, list) corresponding to input signature of the model. If not
      specified, the input model signature is created by
      "build_default_input_model_signature". For example, specify
      "input_model_signature_fn" if an numerical input feature (which is
      consumed as DenseTensorSpec(float32) by default) will be feed differently
      (e.g. RaggedTensor(int64)).
  """

  inspector = inspector_lib.make_inspector(src_path)
  objective = inspector.objective()

  model = CoreModel(
      task=objective.task,
      learner="MANUAL",
      ranking_group=objective.group
      if objective.task == inspector_lib.Task.RANKING else None)

  model._set_from_yggdrasil_model(  # pylint: disable=protected-access
      inspector,
      src_path,
      input_model_signature_fn=input_model_signature_fn)

  model.save(dst_path)


def _list_explicit_arguments(func):
  """Function decorator that adds an "explicit_args" with the explicit args."""

  arguments = inspect.getfullargspec(func)[0]

  def wrapper(*args, **kargs):
    kargs["explicit_args"] = set(
        list(arguments[:len(args)]) + list(kargs.keys()))
    return func(*args, **kargs)

  return wrapper


def _parse_hp_template(template_name) -> Tuple[str, Optional[int]]:
  """Parses a template name as specified by the user.

  Template can versionned:
    "my_template@v5" -> Returns (my_template, 5)
  or non versionned:
    "my_template" -> Returns (my_template, None)

  Args:
    template_name: User specified template.

  Returns:
    Base template name and version.
  """

  malformed_msg = (f"The template \"{template_name}\" is malformed. Expecting "
                   "\"{template}@v{version}\" or \"{template}")

  if "@" in template_name:
    # Template with version.
    parts = template_name.split("@v")
    if len(parts) != 2:
      raise ValueError(malformed_msg)
    base_name = parts[0]
    try:
      version = int(parts[1])
    except:
      raise ValueError(malformed_msg)
    return base_name, version

  else:
    # Template without version?
    return template_name, None


def _get_matching_template(
    template_name: str,
    all_templates: List[HyperParameterTemplate]) -> HyperParameterTemplate:
  """Returns the template that matches a template name.

  Args:
    template_name: User specified template.
    all_templates: Candidate templates.

  Returns:
    The matching template.
  """

  # Extract the base name and version of the template.
  template_base, template_version = _parse_hp_template(template_name)

  if template_version is not None:
    # Template with version.

    # Matching templates.
    matching = [
        template for template in all_templates if
        template.name == template_base and template.version == template_version
    ]

    if not matching:
      available = [
          f"{template.name}@v{template.version}" for template in all_templates
      ]
      raise ValueError(f"No template is matching {template_name}. "
                       f"The available templates are: {available}")

    if len(matching) > 1:
      raise ValueError("Internal error. Multiple matching templates")
    return matching[0]

  else:
    # Template without version?
    matching = [
        template for template in all_templates if template.name == template_base
    ]
    matching.sort(key=lambda x: x.version, reverse=True)

    if not matching:
      available = list(set([template.name for template in all_templates]))
      raise ValueError(f"No template is matching {template_name}. "
                       f"Available template names are: {available}")
    return matching[0]


def _apply_hp_template(parameters: Dict[str, Any], template_name: str,
                       all_templates: List[HyperParameterTemplate],
                       explicit_parameters: Set[str]) -> Dict[str, Any]:
  """Applies the hyper-parameter template to the user+default parameters.

  Look for a template called "template_name" (is "template_name" is a versioned
  template e.g. "name@v5") or for the latest (higher version) template (if
  "template_name" is a non versioned template e.g. "name").

  Once the template is found, merges "parameters" and the template according to
  the user parameters i.e. the final value is (in order of importance):
    user parameters > template parameters > default parameters.

  Args:
    parameters: User and default hyper-parameters.
    template_name: Name of the template as specified by the user.
    all_templates: All the available templates.
    explicit_parameters: Set of parameters (in parameters) defined by the user.

  Returns:
    The merged hyper-parameters.
  """

  template = _get_matching_template(template_name, all_templates)
  logging.info("Resolve hyper-parameter template \"%s\" to \"%s@v%d\" -> %s.",
               template_name, template.name, template.version,
               template.parameters)

  for key in list(parameters.keys()):
    if key in template.parameters and key not in explicit_parameters:
      parameters[key] = template.parameters[key]

  return parameters


def _check_feature_names(feature_names: List[str], raise_error: bool):
  """Checks if the features names are compatible with all of the Keras API."""

  def problem(reason):
    full_reason = (
        "One or more feature names are not compatible with the Keras API: "
        f"{reason} This problem can be solved in one of two ways: (1; "
        "Recommended) Rename the features to be compatible. You can use "
        "the argument `fix_feature_names=True` if you are using "
        "`pd_dataframe_to_tf_dataset`. (2) Disable this error message "
        "(`fail_on_non_keras_compatible_feature_name=False`) and only use part"
        " of the compatible Keras API.")
    if raise_error:
      raise ValueError(full_reason)
    else:
      logging.warning(full_reason)

  # List of character forbidden in a serving signature name.
  for feature_name in feature_names:
    if not feature_name:
      problem("One of the feature names is empty.")

    for character in _FORBIDDEN_FEATURE_CHARACTERS:
      if character in feature_name:
        problem(f"The feature name \"{feature_name}\" contains a forbidden "
                "character ({_FORBIDDEN_FEATURE_CHARACTERS}).")


def _check_dataset(x: tf.data.Dataset):
  """Checks that the dataset is well configured for TF-DF.

  Raise an exception otherwise.

  Args:
    x: A tf.data.Dataset.
  """

  def error(message):
    message += (" Alternatively, you can disabled this check with the "
                "constructor argument `check_dataset=False`. If this message "
                "is a false positive, please let us know so we can improve "
                "this dataset check logic.")
    if ONLY_WARN_ON_DATASET_CONFIGURATION_ISSUES:
      message += (
          " This warning will be turned into an error on "
          "[18 Jan. 2022]. Make sure to solve this issue before this date.")
      logging.warning("%s", message)
    else:
      raise ValueError(message)

  if _contains_repeat(x):
    error(
        "The dataset contain a 'repeat' operation. For maximum quality, TF-DF "
        "models should be trained without repeat operations as the dataset "
        "should only be read once. Remove the repeat operations to solve this "
        "issue.")

  if _contains_shuffle(x):
    error(
        "The dataset contain a 'shuffle' operation. For maximum quality, TF-DF "
        "models should be trained without shuffle operations to make the "
        "algorithm deterministic. To make the the algorithm non deterministic, "
        "change the `random_seed` constructor argument instead. Remove the "
        "shuffle operations to solve this issue.")

  batch_size = _contains_batch(x)
  if batch_size is None:
    error("The dataset does not contain a 'batch' operation. TF-DF models "
          "should be trained without batch operations. Add a batch operations "
          "to solve this issue.")

  num_examples = None
  try:
    num_examples = x.cardinality().numpy() * batch_size
  except:  # pylint: disable=bare-except
    pass

  if (num_examples is None or
      num_examples > 2000) and batch_size is not None and batch_size < 100:
    error(f"The dataset has a small batch size ({batch_size}). TF-DF model "
          "quality is not impacted by the batch size. However, a small batch "
          "size slow down the dataset reading and training preparation. Use "
          "a batch size of at least 100 (1000 if even better) for a "
          "dataset with more than 2k examples to solve this issue.")


def _contains_repeat(dataset) -> bool:
  """Tests if a dataset contains a "repeat()" operation."""

  try:
    if not isinstance(dataset, tf.data.Dataset):
      return False
    if dataset.__class__.__name__ == "RepeatDataset":
      return True
    if hasattr(dataset, "_input_dataset"):
      return _contains_repeat(dataset._input_dataset)  # pylint: disable=protected-access
  except:  # pylint: disable=bare-except
    pass
  return False


def _contains_shuffle(dataset) -> bool:
  """Tests if a dataset contains a "shuffle()" operation."""

  try:
    if not isinstance(dataset, tf.data.Dataset):
      return False
    if dataset.__class__.__name__ == "ShuffleDataset":
      return True
    if hasattr(dataset, "_input_dataset"):
      return _contains_repeat(dataset._input_dataset)  # pylint: disable=protected-access
  except:  # pylint: disable=bare-except
    pass
  return False


def _contains_batch(dataset) -> Optional[int]:
  """Tests if a dataset contains a "batch()" operation.

  Args:
    dataset: A tf.data.Dataset.

  Returns:
    The batch size, or None if not batch operation was found.
  """

  try:
    if not isinstance(dataset, tf.data.Dataset):
      return None
    if dataset.__class__.__name__ == "BatchDataset":
      return dataset._batch_size  # pylint: disable=protected-access
    if hasattr(dataset, "_input_dataset"):
      return _contains_batch(dataset._input_dataset)  # pylint: disable=protected-access
  except:  # pylint: disable=bare-except
    pass
  return None


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


# pylint: enable=g-doc-args
# pylint: enable=g-doc-return-or-yield
