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

import copy
from datetime import datetime  # pylint: disable=g-importing-member
import inspect
import os
import tempfile
from typing import Optional, List, Dict, Any, Text, Tuple, NamedTuple, Set

import tensorflow as tf

from tensorflow.python.data.ops import dataset_ops
from tensorflow_decision_forests.component.inspector import inspector as inspector_lib
from tensorflow_decision_forests.component.tuner import tuner as tuner_lib
from tensorflow_decision_forests.keras import core_inference
from tensorflow_decision_forests.tensorflow import core as tf_core
from tensorflow_decision_forests.tensorflow import tf1_compatibility
from tensorflow_decision_forests.tensorflow import tf_logging
from tensorflow_decision_forests.tensorflow.ops.inference import api as tf_op
from tensorflow_decision_forests.tensorflow.ops.training import op as training_op
from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.learner import abstract_learner_pb2
from yggdrasil_decision_forests.model import abstract_model_pb2  # pylint: disable=unused-import
from yggdrasil_decision_forests.utils.distribute.implementations.grpc import grpc_pb2  # pylint: disable=unused-import

import keras.engine.data_adapter as data_adapter
get_data_handler = data_adapter.get_data_handler

layers = tf.keras.layers
models = tf.keras.models
optimizers = tf.keras.optimizers
losses = tf.keras.losses
backend = tf.keras.backend

no_automatic_dependency_tracking = tf1_compatibility.no_automatic_dependency_tracking

# Hyper-parameters of the model represented as a dictionary of <parameter names,
# parameter values>.
HyperParameters = tf_core.HyperParameters

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
ONLY_WARN_ON_DATASET_CONFIGURATION_ISSUES = False

# Imports from the inference only code.
# pylint: disable=protected-access
Task = core_inference.Task
TaskType = core_inference.TaskType
FeatureColumn = core_inference.FeatureColumn
FeatureSemantic = core_inference.FeatureSemantic
InferenceCoreModel = core_inference.InferenceCoreModel
AdvancedArguments = core_inference.AdvancedArguments
pd_dataframe_to_tf_dataset = core_inference.pd_dataframe_to_tf_dataset
yggdrasil_model_to_keras_model = core_inference.yggdrasil_model_to_keras_model
YggdrasilTrainingConfig = core_inference.YggdrasilTrainingConfig
YggdrasilDeploymentConfig = core_inference.YggdrasilDeploymentConfig
_LABEL = core_inference._LABEL
_RANK_GROUP = core_inference._RANK_GROUP
_UPLIFT_TREATMENT = core_inference._UPLIFT_TREATMENT
_WEIGHTS = core_inference._WEIGHTS
_FORBIDDEN_FEATURE_CHARACTERS = core_inference._FORBIDDEN_FEATURE_CHARACTERS
# pylint: enable=protected-access


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
    num_discretized_numerical_bins: For DISCRETIZED_NUMERICAL features only.
      Number of bins used to discretize DISCRETIZED_NUMERICAL features.
    max_vocab_count: For CATEGORICAL and CATEGORICAL_SET features only. Number
      of unique categorical values stored as string. If more categorical values
      are present, the least frequent values are grouped into a
      Out-of-vocabulary item. Reducing the value can improve or hurt the model.
  """

  def __init__(self,
               name: Text,
               semantic: Optional[FeatureSemantic] = None,
               num_discretized_numerical_bins: Optional[int] = None,
               max_vocab_count: Optional[int] = None):

    self._name = name
    self._semantic = semantic
    self._guide = data_spec_pb2.ColumnGuide()

    # Check matching between hyper-parameters and semantic.
    if semantic != FeatureSemantic.DISCRETIZED_NUMERICAL:
      if num_discretized_numerical_bins is not None:
        raise ValueError("\"discretized\" only works for DISCRETIZED_NUMERICAL"
                         " semantic.")

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
      self._guide.type = data_spec_pb2.NUMERICAL
    elif semantic == FeatureSemantic.DISCRETIZED_NUMERICAL:
      self._guide.type = data_spec_pb2.DISCRETIZED_NUMERICAL
      if num_discretized_numerical_bins is not None:
        self._guide.discretized_numerical.maximum_num_bins = num_discretized_numerical_bins
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


class CoreModel(InferenceCoreModel):
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

  # ...

  # Load a model: it loads as a generic keras model.
  model = tf.keras.models.load_model("/tmp/my_saved_model")
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
      hyper-parameters is available at https://github.com/google/yggdrasil-decision-forests/blob/main/documentation/learners.md.
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
    uplift_treatment: Only for task=Task.CATEGORICAL_UPLIFT and task=Task.
      NUMERICAL_UPLIFT. Name of an integer feature that identifies the treatment
      in an uplift problem. The value 0 is reserved for the control treatment.
    temp_directory: Temporary directory used to store the model Assets after the
      training, and possibly as a work directory during the training. This
      temporary directory is necessary for the model to be exported after
      training e.g. `model.save(path)`. If not specified, `temp_directory` is
      set to a temporary directory using `tempfile.TemporaryDirectory`. This
      directory is deleted when the model python object is garbage-collected.
    verbose: Verbosity mode. 0 = silent, 1 = small details, 2 = full details.
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
      was interrupted by the user (e.g. ctrl+c or "stop" button in a notebook).
      (2) the training job was interrupted (e.g. rescheduling), ond (3) the
      hyper-parameter of the model were changed such that an initially completed
      training is now incomplete (e.g. increasing the number of trees).
      Note: Training can only be resumed if the training datasets is exactly the
        same (i.e. no reshuffle in the tf.data.Dataset).
    check_dataset: If set to true, test if the dataset is well configured for
      the training: (1) Check if the dataset does contains any `repeat`
      operations, (2) Check if the dataset does contain a `batch` operation, (3)
      Check if the dataset has a large enough batch size (min 100 if the dataset
      contains more than 1k examples or if the number of examples is not
      available) If set to false, do not run any test.
    tuner: If set, automatically optimize the hyperparameters of the model using
      this tuner. If the model is trained with distribution (i.e. the model
      definition is wrapper in a TF Distribution strategy, the tuning is
      distributed.
    discretize_numerical_features: If true, discretize all the numerical
      features before training. Discretized numerical features are faster to
      train with, but they can have a negative impact on the model quality.
      Using discretize_numerical_features=True is equivalent as setting the
      feature semantic DISCRETIZED_NUMERICAL in the `feature` argument. See the
      definition of DISCRETIZED_NUMERICAL for more details.
    num_discretize_numerical_bins: Number of bins used when disretizing
      numerical features. The value `num_discretized_numerical_bins` defined in
      a `FeatureUsage` (if any) takes precedence.
  """

  def __init__(
      self,
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
      verbose: int = 1,
      advanced_arguments: Optional[AdvancedArguments] = None,
      num_threads: Optional[int] = None,
      name: Optional[str] = None,
      max_vocab_count: Optional[int] = 2000,
      try_resume_training: Optional[bool] = True,
      check_dataset: Optional[bool] = True,
      tuner: Optional[tuner_lib.Tuner] = None,
      discretize_numerical_features: bool = False,
      num_discretized_numerical_bins: int = 255,
  ) -> None:
    super(CoreModel, self).__init__(
        name=name,
        task=task,
        ranking_group=ranking_group,
        verbose=verbose,
        advanced_arguments=advanced_arguments,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        uplift_treatment=uplift_treatment,
        temp_directory=temp_directory,
    )

    self._learner = learner  # Only used for user inspection
    self._learner_params = learner_params
    self._features = features or []
    self._exclude_non_specified = exclude_non_specified_features
    self._num_threads = num_threads
    self._max_vocab_count = max_vocab_count
    self._try_resume_training = try_resume_training
    self._check_dataset = check_dataset
    self._tuner = tuner
    self._discretize_numerical_features = discretize_numerical_features
    self._num_discretized_numerical_bins = num_discretized_numerical_bins

    # Number of examples. Populated during training
    self._num_training_examples = None
    self._num_validation_examples = None

    # Determine the optimal number of threads.
    if self._num_threads is None:
      self._num_threads = os.cpu_count()
      if self._num_threads is None:
        if self._verbose >= 1:
          tf_logging.warning(
              "Cannot determine the number of CPUs. Set num_threads=6")
        self._num_threads = 6
      else:
        if self._num_threads >= 32:
          if self._verbose >= 1:
            tf_logging.warning(
                "The `num_threads` constructor argument is not set and the "
                "number of CPU is os.cpu_count()=%d > 32. Setting num_threads "
                "to 32. Set num_threads manually to use more than 32 cpus." %
                self._num_threads)
          self._num_threads = 32
        else:
          if self._verbose >= 2:
            tf_logging.info("Use %d thread(s) for training", self._num_threads)

    # Configure the training config.
    if tuner is not None:
      tuner.set_base_learner(learner)
      self._advanced_arguments.yggdrasil_training_config.MergeFrom(
          tuner.train_config())
    else:
      self._advanced_arguments.yggdrasil_training_config.learner = learner

    if not self._features and exclude_non_specified_features:
      raise ValueError(
          "The model does not have any input features: "
          "exclude_non_specified_features is True and not features are "
          "provided as input.")

    if self._temp_directory is None:
      self._temp_directory_handle = tempfile.TemporaryDirectory()
      self._temp_directory = self._temp_directory_handle.name
      if self._verbose >= 1:
        tf_logging.info("Use %s as temporary training directory",
                        self._temp_directory)

    # If the model is trained with weights.
    self._weighted_training = False

    # True if the user provides a validation dataset to `fit`.
    self._has_validation_dataset = False

  @property
  def learner_params(self) -> Optional[HyperParameters]:
    """Gets the dictionary of hyper-parameters passed in the model constructor.

    Changing this dictionary will impact the training.
    """
    return self._learner_params

  @property
  def learner(self) -> Optional[str]:
    """Name of the learning algorithm used to train the model."""
    return self._learner

  @property
  def num_threads(self) -> Optional[int]:
    """Number of threads used to train the model."""
    return self._num_threads

  @property
  def num_training_examples(self) -> Optional[int]:
    """Number of training examples."""
    return self._num_training_examples

  @property
  def num_validation_examples(self) -> Optional[int]:
    """Number of validation examples."""
    return self._num_validation_examples

  @property
  def exclude_non_specified_features(self) -> Optional[bool]:
    """If true, only use the features specified in "features"."""
    return self._exclude_non_specified

  @property
  def training_model_id(self) -> str:
    """Identifier of the model."""
    return self._training_model_id

  def load_weights(self, *args, **kwargs):  # pylint: disable=useless-super-delegation
    """No-op for TensorFlow Decision Forests models.

    `load_weights` is not supported by TensorFlow Decision Forests models.
    To save and restore a model, use the SavedModel API i.e.
    `model.save(...)` and `tf.keras.models.load_model(...)`. To resume the
    training of an existing model, create the model with
    `try_resume_training=True` (default value) and with a similar
    `temp_directory` argument. See documentation of `try_resume_training`
    for more details.

    Args:
      *args: Passed through to base `keras.Model` implemenation.
      **kwargs: Passed through to base `keras.Model` implemenation.
    """
    super(CoreModel, self).load_weights(*args, **kwargs)

  # This function should not be serialized in the SavedModel.
  @no_automatic_dependency_tracking
  @tf.function(reduce_retracing=True)
  def valid_step(self, data):
    """Collects validation examples."""

    return self.collect_data_step(data, is_training_example=False)

  # This function should not be serialized in the SavedModel.
  @no_automatic_dependency_tracking
  @tf.function(reduce_retracing=True)
  def train_step(self, data):
    """Collects training examples."""

    return self.collect_data_step(data, is_training_example=True)

  # This function should not be serialized in the SavedModel.
  @no_automatic_dependency_tracking
  @tf.function(reduce_retracing=True)
  def collect_data_step(self, data, is_training_example):
    """Collect examples e.g. training or validation."""

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

    if self._verbose >= 2:
      tf_logging.info(
          "%s tensor examples:\nFeatures: %s\nLabel: %s\nWeights: %s",
          "Training" if is_training_example else "Validation", train_x, train_y,
          train_weights)

    if isinstance(train_x, dict):
      _check_feature_names(
          train_x.keys(),
          self._advanced_arguments.fail_on_non_keras_compatible_feature_name)

    if self._preprocessing is not None:
      train_x = self._preprocessing(train_x)
      if self._verbose >= 2:
        tf_logging.info("Tensor example after pre-processing:\n%s", train_x)
      if isinstance(train_x, list) and self._features:
        tf_logging.warning(
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
      if self._verbose >= 2:
        tf_logging.info(
            "Squeeze label' shape from [batch_size, 1] to [batch_size]")
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
            "The training weights tensor is expected to be a tensor. "
            f"Got {train_weights} instead.")

      if len(train_weights.shape) != 1:
        if self._verbose >= 2:
          tf_logging.info(
              "Squeeze weight' shape from [batch_size, 1] to [batch_size]")
        train_weights = tf.squeeze(train_weights, axis=1)

      if len(train_weights.shape) != 1:
        raise ValueError(
            "Weights can either be passed in as [batch_size, 1] or [batch_size]. "
            "Invalid shape %s." % train_weights.shape)

    # List the input features and their semantics.
    semantics = tf_core.infer_semantic(
        train_x, {feature.name: feature.semantic for feature in self._features},
        self._exclude_non_specified)

    # The ranking group and treatment are not part of the features unless
    # specified explicitly.
    if (self._ranking_group is not None and
        self._ranking_group not in self._features and
        self._ranking_group in semantics):
      del semantics[self._ranking_group]

    if (self._uplift_treatment is not None and
        self._uplift_treatment not in self._features and
        self._uplift_treatment in semantics):
      del semantics[self._uplift_treatment]

    if is_training_example:
      if self._semantics is not None:
        raise ValueError("The model is already trained")
      self._semantics = semantics
    else:
      self._has_validation_dataset = True
      if self._semantics is None:
        raise ValueError(
            "The validation should be collected after the training")

      if semantics != self._semantics:
        raise ValueError(
            "The validation dataset does not have the same "
            "semantic as the training dataset.\nTraining:\n{}\nValidation:\n{}"
            .format(self._semantics, semantics))

    semantic_inputs = tf_core.combine_tensors_and_semantics(train_x, semantics)

    normalized_semantic_inputs = tf_core.normalize_inputs(
        semantic_inputs,
        categorical_integer_offset_correction=not self._advanced_arguments
        .disable_categorical_integer_offset_correction)

    if self._verbose >= 2:
      tf_logging.info("Normalized tensor features:\n %s",
                      normalized_semantic_inputs)

    if is_training_example:
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

    elif self._task == Task.NUMERICAL_UPLIFT:
      normalized_semantic_inputs[_LABEL] = tf_core.SemanticTensor(
          tensor=tf.cast(train_y, tf_core.NormalizedNumericalType),
          semantic=tf_core.Semantic.NUMERICAL)

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
        tf_core.collect_training_examples(
            normalized_semantic_inputs,
            self._training_model_id,
            collect_training_data=is_training_example)

      else:

        if not is_training_example:
          tf_logging.warning(
              "The validation dataset given to `fit` is not used to help "
              "training (e.g. early stopping) in the case of distributed "
              "training. If you want to use a validation dataset use "
              "non-distributed training or use `fit_from_file` instead.")

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

    # Number of scanned examples.
    return tf.shape(train_y)[0]

  def _distributed_partial_dataset_cache_path(self):
    """Directory accessible from all workers containing the partial cache."""

    return os.path.join(self._temp_directory, "partial_dataset_cache")

  def fit(self,
          x=None,
          y=None,
          callbacks=None,
          verbose: Optional[Any] = None,
          validation_steps: Optional[int] = None,
          validation_data: Optional[Any] = None,
          sample_weight: Optional[Any] = None,
          steps_per_epoch: Optional[Any] = None,
          class_weight: Optional[Any] = None,
          **kwargs) -> tf.keras.callbacks.History:
    """Trains the model.

    Local training
    ==============

    It is recommended to use a Pandas Dataframe dataset and to convert it to
    a TensorFlow dataset with "pd_dataframe_to_tf_dataset()":

      pd_dataset = pandas.Dataframe(...)
      tf_dataset = pd_dataframe_to_tf_dataset(dataset, label="my_label")
      model.fit(pd_dataset)

    The following dataset formats are supported:

      1. "x" is a tf.data.Dataset containing a tuple "(features, labels)".
         "features" can be a dictionary a tensor, a list of tensors or a
         dictionary of tensors (recommended). "labels" is a tensor.

      2. "x" is a tensor, list of tensors or dictionary of tensors containing
         the input features. "y" is a tensor.

      3. "x" is a numpy-array, list of numpy-arrays or dictionary of
         numpy-arrays containing the input features. "y" is a numpy-array.

    IMPORTANT: This model trains on the entire dataset at once. This has the
    following consequences:

      1. The dataset need to be read exactly once. If you use a TensorFlow
         dataset, make sure NOT to add a "repeat" operation.
      2. The algorithm does not benefit from shuffling the dataset. If you use a
         TensorFlow dataset, make sure NOT to add a "shuffle" operation.
      3. The dataset needs to be batched (i.e. with a "batch" operation).
         However, the number of elements per batch has not impact on the model.
         Generally, it is recommended to use batches as large as possible as its
         speeds-up reading the dataset in TensorFlow.

    Input features do not need to be normalized (e.g. dividing numerical values
    by the variance) or indexed (e.g. replacing categorical string values by
    an integer). Additionnaly, missing values can be consumed natively.

    Distributed training
    ====================

    Some of the learning algorithms will support distributed training with the
    ParameterServerStrategy.

    In this case, the dataset is read asynchronously in between the workers. The
    distribution of the training depends on the learning algorithm.

    Like for non-distributed training, the dataset should be read eactly once.
    The simplest solution is to divide the dataset into different files (i.e.
    shards) and have each of the worker read a non overlapping subset of shards.

    IMPORTANT: The training dataset should not be infinite i.e. the training
    dataset should not contain any repeat operation.

    Currently (to be changed), the validation dataset (if provided) is simply
    feed to the "model.evaluate()" method. Therefore, it should satify Keras'
    evaluate API. Notably, for distributed training, the validation dataset
    should be infinite (i.e. have a repeat operation).

    See https://www.tensorflow.org/decision_forests/distributed_training for
    more details and examples.

    Here is a single example of distributed training using PSS for both dataset
    reading and training distribution.

      def dataset_fn(context, paths, training=True):
        ds_path = tf.data.Dataset.from_tensor_slices(paths)


        if context is not None:
          # Train on at least 2 workers.
          current_worker = tfdf.keras.get_worker_idx_and_num_workers(context)
          assert current_worker.num_workers > 2

          # Split the dataset's examples among the workers.
          ds_path = ds_path.shard(
              num_shards=current_worker.num_workers,
              index=current_worker.worker_idx)

        def read_csv_file(path):
          numerical = tf.constant([math.nan], dtype=tf.float32)
          categorical_string = tf.constant([""], dtype=tf.string)
          csv_columns = [
              numerical,  # age
              categorical_string,  # workclass
              numerical,  # fnlwgt
              ...
          ]
          column_names = [
            "age", "workclass", "fnlwgt", ...
          ]
          label_name = "label"
          return tf.data.experimental.CsvDataset(path, csv_columns, header=True)

        ds_columns = ds_path.interleave(read_csv_file)

        def map_features(*columns):
          assert len(column_names) == len(columns)
          features = {column_names[i]: col for i, col in enumerate(columns)}
          label = label_table.lookup(features.pop(label_name))
          return features, label

        ds_dataset = ds_columns.map(map_features)
        if not training:
          dataset = dataset.repeat(None)
        ds_dataset = ds_dataset.batch(batch_size)
        return ds_dataset

      strategy = tf.distribute.experimental.ParameterServerStrategy(...)
      sharded_train_paths = [list of dataset files]
      with strategy.scope():
        model = DistributedGradientBoostedTreesModel()
        train_dataset = strategy.distribute_datasets_from_function(
          lambda context: dataset_fn(context, sharded_train_paths))

        test_dataset = strategy.distribute_datasets_from_function(
          lambda context: dataset_fn(context, sharded_test_paths))

      model.fit(sharded_train_paths)
      evaluation = model.evaluate(test_dataset, steps=num_test_examples //
        batch_size)

    Args:
      x: Training dataset (See details above for the supported formats).
      y: Label of the training dataset. Only used if "x" does not contains the
        labels.
      callbacks: Callbacks triggered during the training. The training runs in a
        single epoch, itself run in a single step. Therefore, callback logic can
        be called equivalently before/after the fit function.
      verbose: Verbosity mode. 0 = silent, 1 = small details, 2 = full details.
      validation_steps: Number of steps in the evaluation dataset when
        evaluating the trained model with model.evaluate(). If not specified,
        evaluates the model on the entire dataset (generally recommended; not
        yet supported for distributed datasets).
      validation_data: Validation dataset. If specified, the learner might use
        this dataset to help training e.g. early stopping.
      sample_weight: Training weights. Note: training weights can also be
        provided as the third output in a tf.data.Dataset e.g. (features, label,
        weights).
      steps_per_epoch: [Parameter will be removed] Number of training batch to
        load before training the model. Currently, only supported for
        distributed training.
      class_weight: For binary classification only. Mapping class indices
        (integers) to a weight (float) value. Only available for non-Distributed
        training. For maximum compatibility, feed example weights through the
        tf.data.Dataset or using the `weight` argument of
        pd_dataframe_to_tf_dataset.
      **kwargs: Extra arguments passed to the core keras model's fit. Note that
        not all keras' model fit arguments are supported.

    Returns:
      A `History` object. Its `History.history` attribute is not yet
      implemented for decision forests algorithms, and will return empty.
      All other fields are filled as usual for `Keras.Mode.fit()`.
    """

    self._clear_function_cache()

    # Override the verbose
    if verbose is not None:
      self._verbose = verbose

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

    for extra_key, extra_values in kwargs.items():
      if extra_key == "epochs":
        # TODO: Remove epoch argument.
        if extra_values != 1:
          raise ValueError(
              "all decision forests algorithms train with only 1 " +
              "epoch, epochs={} given".format(kwargs["epochs"]))
      else:
        tf_logging.warning(
            "Model constructor argument %s=%s not supported. "
            "See https://www.tensorflow.org/decision_forests/migration for an "
            "explanation about the specificities of TF-DF.", extra_key,
            extra_values)

    if steps_per_epoch is not None:
      tf_logging.warning(
          "The steps_per_epoch argument is deprecated. Instead, feed a "
          "finite dataset (i.e. a dataset without repeat statement) and remove "
          "the steps_per_epoch argument.")

    # Reset the training status.
    self._is_trained.assign(False)

    return self._fit_implementation(
        x=x,
        y=y,
        callbacks=callbacks,
        verbose=verbose,
        validation_steps=validation_steps,
        validation_data=validation_data,
        sample_weight=sample_weight,
        steps_per_epoch=steps_per_epoch,
        class_weight=class_weight)

  @no_automatic_dependency_tracking
  @tf.function(reduce_retracing=True)
  def _consumes_training_examples_until_eof(self, iterator):
    """Consumes all the available training examples.

    The examples are either loaded in memory (local training) or indexed
    (distributed training). Returns the total number of consumed examples.

    Args:
      iterator: Iterator of examples.

    Returns:
      Number of read examples.
    """

    num_examples = 0
    for data in iterator:
      num_examples += self.train_step(data)
    return num_examples

  @no_automatic_dependency_tracking
  @tf.function(reduce_retracing=True)
  def _consumes_validation_examples_until_eof(self, iterator):
    """Consumes all the available validation examples.

    The examples are either loaded in memory (local training) or indexed
    (distributed training). Returns the total number of consumed examples.

    Args:
      iterator: Iterator of examples.

    Returns:
      Number of read examples.
    """

    num_examples = 0
    for data in iterator:
      num_examples += self.valid_step(data)
    return num_examples

  def _fit_implementation(
      self, x, y, verbose, callbacks, sample_weight, validation_data,
      validation_steps, steps_per_epoch, class_weight
  ) -> tf.keras.callbacks.History:  # pylint: disable=g-doc-args,g-doc-return-or-yield
    """Train the model.

    This method performs operations that resembles as the Keras' fit function.

    Details:
      1. Load the training dataset in Yggdrasil (locally or remotely).
      2. Load the validation dataset in Yggdrasil (locally or ignored).
      3. Train the Yggdrasil model. Depending on the learner, this might also
        evaluate the model (on the validation dataset or with out-of-bag).
      4. Evaluate the model using the Keras metrics.

      We recommend training / validating models with finite datasets. For
      backward compatibility with the early version of TF-DF, we also support
      infinite dataset with specified number of steps (steps_per_epoch and
      validation_steps arguments). Using a number of steps currently raises
      a warning, and will later raise an error (and the logic be removed).
    """

    # Create the callback manager
    if not isinstance(callbacks, tf.keras.callbacks.CallbackList):
      callbacks = tf.keras.callbacks.CallbackList(
          callbacks, model=self, add_history=False)

    # Manages the history manually.
    # Note: The both the History and CallbackList object will override the
    # "model.history" field.
    history = tf.keras.callbacks.History()
    history.model = self
    history.on_train_begin()
    history.on_epoch_begin(0)

    callbacks.on_train_begin()
    callbacks.on_epoch_begin(0)

    # Check if the training is local or distributed.
    distribution_config = tf_core.get_distribution_configuration(
        self.distribute_strategy)

    # Instantiate the validation data handler.
    # We create the handler before the training such that if it is
    # misconfigured, the error is raised immediately (instead of after
    # training).
    validation_data_handler = None
    if validation_data:
      val_x, val_y, val_sample_weight = (
          tf.keras.utils.unpack_x_y_sample_weight(validation_data))

      if distribution_config is None:
        # TODO: The validation_data is currently not used for distributed
        # training (except for the model validation evaluation). Creating
        # and not using the validation_data_handler for distributed training
        # seems to cause some issues.

        # Create data_handler for evaluation and cache it.
        validation_data_handler = get_data_handler(
            x=val_x,
            y=val_y,
            sample_weight=val_sample_weight,
            model=self,
            steps_per_epoch=validation_steps,
            class_weight=class_weight)

    time_begin_train_dataset_reading = datetime.now()
    if self._verbose >= 1:
      tf_logging.info("Reading training dataset...")

    coordinator = None
    if distribution_config is None:
      # Local training.

      # Wraps the input training dataset into a tf.data.Dataset like object.
      # This is a noop if the training dataset is already provided as a
      # tf.data.Dataset.
      data_handler = get_data_handler(
          x=x,
          y=y,
          sample_weight=sample_weight,
          model=self,
          class_weight=class_weight)
      iterator = iter(data_handler._dataset)  # pylint: disable=protected-access

      if steps_per_epoch is None:
        # Local training with finite dataset.

        # Load the entire training dataset in Yggdrasil in a single TensorFlow
        # step.
        self._num_training_examples = self._consumes_training_examples_until_eof(
            iterator)

      else:
        # Local training with number of steps.

        # TODO: Make this case an error and remove this code.
        tf_logging.warning(
            "You are using non-distributed training with steps_per_epoch. "
            "This solution will lead to a sub-optimal model. Instead, "
            "use a finite training dataset (e.g. a dataset without "
            "repeat operation) and remove the `steps_per_epoch` argument. "
            "This warning will be turned into an error in the future.")

        def consumes_one_training_batch(iterator):
          data = next(iterator)
          return self.train_step(data)

        tf_consumes_one_training_batch = tf.function(
            consumes_one_training_batch, reduce_retracing=True)

        num_examples = 0
        try:
          for _ in range(steps_per_epoch):
            num_examples += tf_consumes_one_training_batch(iterator)
        except tf.errors.OutOfRangeError:
          pass
        self._num_training_examples = num_examples

        # End of the logic to remove as mentionned in the TODO above.

    else:

      if class_weight is not None:
        raise ValueError("class_weight not support for distributed training. "
                         "Feed the example weights through the dataset.")

      # Distributed training
      if not self.capabilities().support_partial_cache_dataset_format:
        raise ValueError(
            f"The model {type(self)} does not support training with a TF "
            "Distribution strategy (i.e. model.capabilities()."
            "support_partial_cache_dataset_format == False). If the dataset "
            "is small, simply remove the distribution strategy scope (i.e. `with "
            "strategy.scope():` around the model construction). If the dataset "
            "is large, use a distributed version of the model. For example, "
            "use DistributedGradientBoostedTreesModel instead of "
            "GradientBoostedTreesModel.")

      # Make the workers read the entire dataset.
      # If a worker is interrupted, it restarts reading its part of the dataset
      # from the start.
      coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
          self.distribute_strategy)
      with self.distribute_strategy.scope():
        per_worker_dataset = coordinator.create_per_worker_dataset(x)
      per_worker_iter = iter(per_worker_dataset)

      if steps_per_epoch is not None:
        # Distributed training with number of steps.

        # TODO: Deprecated logic. To remove.

        tf_logging.warning(
            "You are using distributed training with steps_per_epoch. "
            "This solution will lead to a sub-optimal model. Instead, "
            "use a finite training dataset (e.g. a dataset without "
            "repeat operation) and remove the `steps_per_epoch` argument.")

        def remote_consumes_one_training_batch(iterator):
          data = next(iterator)
          self.train_step(data)

        tf_remote_consumes_one_training_batch = tf.function(
            remote_consumes_one_training_batch, reduce_retracing=True)

        tf_logging.info("Scheduling of %d steps", steps_per_epoch)
        for _ in range(steps_per_epoch):
          coordinator.schedule(
              tf_remote_consumes_one_training_batch, args=(per_worker_iter,))
        tf_logging.info("Waiting for scheduled steps to complete")
        coordinator.join()

        # The number of training examples is unknown.
        self._num_training_examples = -1

        # End of the logic to remove as mentionned in the TODO above.

      else:
        # Distributed training with finite dataset.

        def remote_consumes_training_examples_until_eof(iterator):
          return self._consumes_training_examples_until_eof(iterator)

        tf_remote_consumes_training_examples_until_eof = tf.function(
            remote_consumes_training_examples_until_eof, reduce_retracing=True)

        # TODO: Replace with an coordinator.schedule version when
        # available.
        self._num_training_examples = tf_core.execute_function_on_each_worker(
            coordinator, tf_remote_consumes_training_examples_until_eof,
            (per_worker_iter,))

    if self._verbose >= 1:
      tf_logging.info("Training dataset read in %s. Found %d examples.",
                      datetime.now() - time_begin_train_dataset_reading,
                      self._num_training_examples)

    # Load the validation dataset
    if validation_data is not None:

      # Collect the validation dataset in Yggdrasil.
      if coordinator is not None:

        # TODO: Collect the validation dataset in yggdrasil when using
        # distributed training.
        assert validation_data_handler is None

        tf_logging.warning(
            "With distributed training, the validation dataset is not (yet) "
            "used for early stopping.")
        self._num_validation_examples = None

      else:
        assert validation_data_handler is not None

        time_begin_valid_dataset_reading = datetime.now()
        if self._verbose >= 1:
          tf_logging.info("Reading validation dataset...")

        # Load the validation examples in Yggdrasil.
        iterator = iter(validation_data_handler._dataset)  # pylint: disable=protected-access

        if validation_steps is None:
          # Validation with finite dataset.

          self._num_validation_examples = self._consumes_validation_examples_until_eof(
              iterator)

        else:

          # Validation with number of steps.
          # TODO: Make this case an error and remove this code.
          tf_logging.warning(
              "You are using non-distributed validation with steps_per_epoch. "
              "This solution will lead to a sub-optimal model. Instead, "
              "use a finite validation dataset (e.g. a dataset without "
              "repeat operation) and remove the `validation_steps` argument. "
              "This warning will be turned into an error in the future.")

          def consumes_one_valid_batch(iterator):
            data = next(iterator)
            return self.valid_step(data)

          tf_consumes_one_valid_batch = tf.function(
              consumes_one_valid_batch, reduce_retracing=True)

          num_examples = 0
          try:
            for _ in range(validation_steps):
              num_examples += tf_consumes_one_valid_batch(iterator)
          except tf.errors.OutOfRangeError:
            pass
          self._num_validation_examples = num_examples

          # End of the logic to remove as mentionned in the TODO above.

        tf_logging.info("Num validation examples: %s",
                        self.num_validation_examples)

        if self._verbose >= 1:
          tf_logging.info("Validation dataset read in %s. Found %d examples.",
                          datetime.now() - time_begin_valid_dataset_reading,
                          self._num_validation_examples)

    # Train the model
    # Note: The model should be trained after having collected both the training
    # and validation datasets.

    time_begin_training_model = datetime.now()
    if self._verbose >= 1:
      tf_logging.info("Training model...")

    self._train_model(cluster_coordinator=coordinator)

    if self._verbose >= 1:
      tf_logging.info("Model trained in %s",
                      datetime.now() - time_begin_training_model)

    # Load the inference ops of the model.
    self._build(x)

    # Evaluate the model using Keras
    if (validation_data is not None and
        not self._advanced_arguments.populate_history_with_yggdrasil_logs):
      # Note: the "validation_data_handler" is exausted and cannot be reused
      # with the model evaluation (i.e. _use_cached_eval_dataset=True).

      # TODO: Even in the case of distributed training, implement an exact
      # evaluation of the model i.e. each validation example is evaluated once
      # and exactly once.history

      # Evaluate the model using Keras metrics.
      val_logs = self.evaluate(
          x=val_x,
          y=val_y,
          verbose="auto" if self._verbose > 0 else 0,
          sample_weight=val_sample_weight,
          return_dict=True,
          steps=validation_steps,
          callbacks=callbacks)

      val_logs = {"val_" + name: val for name, val in val_logs.items()}
      callbacks.on_epoch_end(0, val_logs)
      history.on_epoch_end(0, val_logs)

    else:
      last_logs = self._add_training_logs_to_history(history=history)
      callbacks.on_epoch_end(0, last_logs)
      history.on_epoch_end(0, last_logs)

    callbacks.on_train_end()
    self.history = history
    return self.history

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

    if self._verbose >= 1:
      tf_logging.info("Training model on file dataset %s", train_path)

    self._clear_function_cache()

    # Call "compile" if the user forgot to do so.
    if not self._is_compiled:
      self.compile()

    train_model_path = self._temp_directory
    model_path = os.path.join(train_model_path, "model")

    # Create the dataspec guide.
    guide = data_spec_pb2.DataSpecificationGuide(
        ignore_columns_without_guides=self._exclude_non_specified,
        max_num_scanned_rows_to_accumulate_statistics=max_num_scanned_rows_to_accumulate_statistics,
        detect_numerical_as_discretized_numerical=self
        ._discretize_numerical_features)
    guide.default_column_guide.categorial.max_vocab_count = self._max_vocab_count
    guide.default_column_guide.discretized_numerical.maximum_num_bins = self._num_discretized_numerical_bins

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

    with tf_logging.capture_cpp_log_context(verbose=self._verbose >= 2):
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
          learner="",
          task=self._task,
          generic_hparms=tf_core.hparams_dict_to_generic_proto(
              self._learner_params),
          ranking_group=ranking_key,
          keep_model_in_resource=True,
          guide=guide,
          training_config=self._advanced_arguments.yggdrasil_training_config,
          deployment_config=deployment_config,
          working_cache_path=os.path.join(self._temp_directory,
                                          "working_cache"),
          distribution_config=distribution_config,
          try_resume_training=try_resume_training)

      # Request and store a description of the model.
      self._description = training_op.SimpleMLShowModel(
          model_identifier=self._training_model_id).numpy().decode("utf-8")
      training_op.SimpleMLUnloadModel(model_identifier=self._training_model_id)

    # Build the model's graph.
    inspector = inspector_lib.make_inspector(
        model_path, file_prefix=self._training_model_id)
    self._set_from_yggdrasil_model(
        inspector,
        model_path,
        file_prefix=self._training_model_id,
        input_model_signature_fn=input_model_signature_fn)

    # Build the model history.
    history = self._training_logs_to_history(inspector)
    self.history = history  # Expected by Keras.
    return history

  def _add_training_logs_to_history(
      self,
      history: tf.keras.callbacks.History,
      inspector: Optional[inspector_lib.AbstractInspector] = None
  ) -> Optional[Dict[str, Any]]:
    if inspector is None:
      inspector = self.make_inspector()

    training_logs = inspector.training_logs()
    last_logs = None
    if training_logs is not None:
      for src_logs in training_logs:
        if src_logs.evaluation is not None:
          last_logs = src_logs.evaluation.to_dict()
          # TF-DF does not have the concept of "batches" for training.
          # Instead, we use the "batch/step" counter to represent the Number
          # of "iteration/trees".
          history.on_batch_end(src_logs.num_trees, last_logs)
    return last_logs

  def _training_logs_to_history(
      self,
      inspector: Optional[inspector_lib.AbstractInspector] = None
  ) -> tf.keras.callbacks.History:
    history = tf.keras.callbacks.History()
    history.model = self
    history.on_train_begin()
    history.on_epoch_begin(0)
    last_logs = self._add_training_logs_to_history(
        history=history, inspector=inspector)
    history.on_epoch_end(0, last_logs)
    history.on_train_end()
    return history

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

  @staticmethod
  def predefined_hyperparameters() -> List[HyperParameterTemplate]:
    """Returns a better than default set of hyper-parameters.

    They can be used directly with the `hyperparameter_template` argument of the
    model constructor.

    These hyper-parameters outperform the default hyper-parameters (either
    generally or in specific scenarios). Like default hyper-parameters, existing
    pre-defined hyper-parameters cannot change.
    """

    return []

  def _train_model(self, cluster_coordinator=None):
    """Effectively train the model."""

    if self._normalized_input_keys is None:
      raise Exception("The training graph was not built.")

    train_model_path = self._temp_directory
    model_path = os.path.join(train_model_path, "model")

    # Create the dataspec guide.
    guide = data_spec_pb2.DataSpecificationGuide(
        detect_numerical_as_discretized_numerical=self
        ._discretize_numerical_features)
    guide.default_column_guide.categorial.max_vocab_count = self._max_vocab_count
    guide.default_column_guide.discretized_numerical.maximum_num_bins = self._num_discretized_numerical_bins

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

    with tf_logging.capture_cpp_log_context(verbose=self._verbose >= 2):

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
            learner="",
            task=self._task,
            generic_hparms=tf_core.hparams_dict_to_generic_proto(
                self._learner_params),
            ranking_group=_RANK_GROUP if self._task == Task.RANKING else None,
            uplift_treatment=_UPLIFT_TREATMENT if self._task
            in [Task.CATEGORICAL_UPLIFT, Task.NUMERICAL_UPLIFT] else None,
            keep_model_in_resource=True,
            guide=guide,
            training_config=self._advanced_arguments.yggdrasil_training_config,
            deployment_config=deployment_config,
            try_resume_training=self._try_resume_training,
            has_validation_dataset=self._has_validation_dataset)

      else:
        tf_core.finalize_distributed_dataset_collection(
            cluster_coordinator=cluster_coordinator,
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
            learner="",
            task=self._task,
            generic_hparms=tf_core.hparams_dict_to_generic_proto(
                self._learner_params),
            ranking_group=_RANK_GROUP if self._task == Task.RANKING else None,
            uplift_treatment=_UPLIFT_TREATMENT if self._task
            in [Task.CATEGORICAL_UPLIFT, Task.NUMERICAL_UPLIFT] else None,
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

      # Load and optimize the model in memory.
      # Register the model as a SavedModel asset.
      additional_args = {}
      additional_args["file_prefix"] = self._training_model_id
      self._model = tf_op.ModelV2(
          model_path=model_path, verbose=False, **additional_args)

  @staticmethod
  def capabilities() -> abstract_learner_pb2.LearnerCapabilities:
    """Lists the capabilities of the learning algorithm."""

    return abstract_learner_pb2.LearnerCapabilities()


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
  tf_logging.info(
      "Resolve hyper-parameter template \"%s\" to \"%s@v%d\" -> %s.",
      template_name, template.name, template.version, template.parameters)

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
      tf_logging.warning("%s",full_reason)

  # List of character forbidden in a serving signature name.
  for feature_name in feature_names:
    if not feature_name:
      problem("One of the feature names is empty.")

    for character in _FORBIDDEN_FEATURE_CHARACTERS:
      if character in feature_name:
        problem(f"The feature name \"{feature_name}\" contains a forbidden "
                f"character ({_FORBIDDEN_FEATURE_CHARACTERS!r}).")


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
      tf_logging.warning(message)
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

  batch_size = _get_batch_size(x)
  if batch_size is None:
    error("The dataset does not contain a 'batch' operation. TF-DF models "
          "should be trained with batch operations. Add a batch operations "
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


def _get_operation(dataset, cls):
  """Returns an operation defined by cls if present in dataset or else None."""
  try:
    if not isinstance(dataset, tf.data.Dataset):
      return None
    if isinstance(dataset, cls):
      return dataset
    if hasattr(dataset, "_dataset"):
      operation = _get_operation(dataset._dataset, cls)  # pylint: disable=protected-access
      if operation is not None:
        return operation
    if hasattr(dataset, "_input_dataset"):
      operation = _get_operation(dataset._input_dataset, cls)  # pylint: disable=protected-access
      if operation is not None:
        return operation
    if hasattr(dataset, "_input_datasets"):
      for input_dataset in dataset._input_datasets:  # pylint: disable=protected-access
        operation = _get_operation(input_dataset, cls)
        if operation is not None:
          return operation
  except:  # pylint: disable=bare-except
    pass
  return None


def _contains_repeat(dataset) -> bool:
  """Tests if a dataset contains a "repeat()" operation."""
  return _get_operation(dataset, dataset_ops.RepeatDataset) is not None


def _contains_shuffle(dataset) -> bool:
  """Tests if a dataset contains a "shuffle()" operation."""
  return _get_operation(dataset, dataset_ops.ShuffleDataset) is not None


def _get_batch_size(dataset) -> Optional[int]:
  """Returns batch_size if dataset contains a "batch()" operation or else None.

  Args:
    dataset: A tf.data.Dataset.

  Returns:
    The batch size, or None if not batch operation was found.
  """

  operation = _get_operation(dataset, dataset_ops.BatchDataset)
  if operation is not None and hasattr(operation, "_batch_size"):
    return operation._batch_size  # pylint: disable=protected-access
  return None
