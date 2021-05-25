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
from functools import partial  # pylint: disable=g-importing-member
import inspect
import os
import tempfile
from typing import Optional, List, Dict, Any, Union, Text, Tuple, NamedTuple, Set
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

# Advanced configuration for the underlying learning library.
YggdrasilDeploymentConfig = abstract_learner_pb2.DeploymentConfig
YggdrasilTrainingConfig = abstract_learner_pb2.TrainingConfig


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
    max_vocab_count: For CATEGORICAL features only. Number of unique categorical
      values. If more categorical values are present, the least frequent values
      are grouped into a Out-of-vocabulary item. Reducing the value can improve
      or hurt the model.
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

    if semantic != FeatureSemantic.CATEGORICAL:
      if max_vocab_count is not None:
        raise ValueError(
            "\"max_vocab_count\" only works for CATEGORICAL semantic.")

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
      self._guide.type = data_spec_pb2.CATEGORICAL
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
  """

  infer_prediction_signature: Optional[bool] = True
  yggdrasil_training_config: Optional[YggdrasilTrainingConfig] = None
  yggdrasil_deployment_config: Optional[YggdrasilDeploymentConfig] = None


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
    ranking_group: Only for task=Task.RANKING. Name of a tf.string feature that
      identifies queries in a query/document ranking task. The ranking group is
      not added automatically for the set of features if
      exclude_non_specified_features=false.
    temp_directory: Temporary directory used during the training. The space
      required depends on the learner. In many cases, only a temporary copy of a
      model will be there.
    verbose: If true, displays information about the training.
    advanced_arguments: Advanced control of the model that most users won't need
      to use. See `AdvancedArguments` for details.
  """

  def __init__(self,
               task: Optional[TaskType] = Task.CLASSIFICATION,
               learner: Optional[str] = "RANDOM_FOREST",
               learner_params: Optional[HyperParameters] = None,
               features: Optional[List[FeatureUsage]] = None,
               exclude_non_specified_features: Optional[bool] = False,
               preprocessing: Optional["models.Functional"] = None,
               ranking_group: Optional[str] = None,
               temp_directory: Optional[str] = None,
               verbose: Optional[bool] = True,
               advanced_arguments: Optional[AdvancedArguments] = None) -> None:
    super(CoreModel, self).__init__()

    self._task = task
    self._learner = learner
    self._learner_params = learner_params
    self._features = features or []
    self._exclude_non_specified = exclude_non_specified_features
    self._preprocessing = preprocessing
    self._ranking_group = ranking_group
    self._temp_directory = temp_directory
    self._verbose = verbose

    # Internal, indicates whether the first evaluation during training,
    # triggered by providing validation data, should trigger the training
    # itself.
    self._train_on_evaluate: bool = False

    if advanced_arguments is None:
      advanced_arguments = AdvancedArguments()
    self._advanced_arguments = advanced_arguments

    if not self._features and exclude_non_specified_features:
      raise ValueError(
          "The model does not have any input features: "
          "exclude_non_specified_features is True and not features are "
          "provided as input.")

    if self._temp_directory is None:
      self._temp_directory = tempfile.mkdtemp()
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

    # Semantic of the input features.
    # Also defines what are the input features of the model.
    self._semantics: Optional[Dict[Text, FeatureSemantic]] = None

    # List of Yggdrasil feature identifiers i.e. feature seen by the Yggdrasil
    # learner. Those are computed after the preprocessing, unfolding and
    # casting.
    self._normalized_input_keys: Optional[List[Text]] = None

    # Textual description of the model.
    self._description: Optional[Text] = None

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
      data = _expand_1d(data)
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
    def test_function_trained(iterator, model):
      """Evaluation of a trained model.

      The only difference with "super.make_test_function()" is that
      "self.test_function" is not set.

      Args:
        iterator: Iterator over dataset.
        model: Model object.

      Returns:
        Evaluation metrics.
      """

      def run_step(data):
        outputs = model.test_step(data)
        with tf.control_dependencies(_minimum_control_deps(outputs)):
          model._test_counter.assign_add(1)  # pylint:disable=protected-access
        return outputs

      data = next(iterator)
      return run_step(data)

    if self._is_trained:
      return partial(test_function_trained, model=self)
    else:
      return test_function_not_trained

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
    elif isinstance(inputs, list):
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

    # Apply the model.
    predictions = self._model.apply(normalized_inputs)

    if (self._task == Task.CLASSIFICATION and
        predictions.dense_predictions.shape[1] == 2):
      # Yggdrasil returns the probably of both classes in binary classification.
      # Keras expects only the value (logic or probability) of the "positive"
      # class (value=1).
      return predictions.dense_predictions[:, 1:2]
    else:
      return predictions.dense_predictions

  # This function should not be serialized in the SavedModel.
  @base_tracking.no_automatic_dependency_tracking
  @tf.function(experimental_relax_shapes=True)
  def train_step(self, data):
    """Collects training examples."""

    if isinstance(data, dict):
      raise ValueError("No label received for training. If you used "
                       "`pd_dataframe_to_tf_dataset`, make sure to "
                       f"specify the `label` argument. data={data}")

    train_x, train_y = data
    if self._verbose:
      logging.info("Collect training examples.\nFeatures: %s\nLabel: %s",
                   train_x, train_y)

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
    elif isinstance(train_x, list):
      # Note: The name of a tensor (value.name) can change between the training
      # and the inference.
      train_x = {str(idx): value for idx, value in enumerate(train_x)}
    else:
      raise ValueError(
          f"The training input tensor is expected to be a tensor, list of "
          f"tensors or a dictionary of tensors. Got {train_x} instead")

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

    # List the input features and their semantics.
    assert self._semantics is None, "The model is already trained"
    self._semantics = tf_core.infer_semantic(
        train_x, {feature.name: feature.semantic for feature in self._features},
        self._exclude_non_specified)

    # The ranking group is not part of the features, unless specified
    # explicitly.
    if (self._ranking_group is not None and
        self._ranking_group not in self._features and
        self._ranking_group in self._semantics):
      del self._semantics[self._ranking_group]

    semantic_inputs = tf_core.combine_tensors_and_semantics(
        train_x, self._semantics)

    normalized_semantic_inputs = tf_core.normalize_inputs(semantic_inputs)

    if self._verbose:
      logging.info("Normalized features: %s", normalized_semantic_inputs)

    self._normalized_input_keys = sorted(
        list(normalized_semantic_inputs.keys()))

    # Adds the semantic of the label.
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

    else:
      raise Exception("Non supported task {}".format(self._task))

    if not self._is_trained:
      tf_core.collect_training_examples(normalized_semantic_inputs,
                                        self._training_model_id)

    # Not metrics are returned during the collection of training examples.
    return {}

  def compile(self, metrics=None):
    """Configure the model for training.

    Unlike for most Keras model, calling "compile" is optional before calling
    "fit".

    Args:
      metrics: Metrics to report during training.

    Raises:
      ValueError: Invalid arguments.
    """

    super(CoreModel, self).compile(metrics=metrics)

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

    Pandas Dataframe can be consumed with "dataframe_to_tf_dataset":
      dataset = pandas.Dataframe(...)
      model.fit(pd_dataframe_to_tf_dataset(dataset, label="my_label"))

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

    try:
      history = super(CoreModel, self).fit(
          x=x, y=y, epochs=1, callbacks=callbacks, **kwargs)
    finally:
      self._train_on_evaluate = False

    self._build(x)

    return history

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

    train_model_path = self._temp_directory
    model_path = os.path.join(train_model_path, "model")

    # Create the dataspec guide.
    guide = data_spec_pb2.DataSpecificationGuide()
    for feature in self._features:
      col_guide = copy.deepcopy(feature.guide)
      col_guide.column_name_pattern = tf_core.normalize_inputs_regexp(
          feature.name)
      guide.column_guides.append(col_guide)

    # Train the model.
    # The model will be exported to "train_model_path".
    #
    # Note: It would be possible to train and load the model without saving the
    # model to disk.
    tf_core.train(
        input_ids=self._normalized_input_keys,
        label_id=_LABEL,
        model_id=self._training_model_id,
        model_dir=train_model_path,
        learner=self._learner,
        task=self._task,
        generic_hparms=tf_core.hparams_dict_to_generic_proto(
            self._learner_params),
        ranking_group=_RANK_GROUP if self._task == Task.RANKING else None,
        keep_model_in_resource=True,
        guide=guide,
        training_config=self._advanced_arguments.yggdrasil_training_config,
        deployment_config=self._advanced_arguments.yggdrasil_deployment_config,
    )

    # Request and store a description of the model.
    self._description = training_op.SimpleMLShowModel(
        model_identifier=self._training_model_id).numpy().decode("utf-8")
    training_op.SimpleMLUnloadModel(model_identifier=self._training_model_id)

    self._is_trained.assign(True)

    # Load and optimize the model in memory.
    # Register the model as a SavedModel asset.
    self._model = tf_op.ModelV2(model_path=model_path, verbose=False)

  def _set_from_yggdrasil_model(self,
                                inspector: inspector_lib.AbstractInspector,
                                path: str):

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

    # Creates a toy batch to initialize the Keras model. The values are not
    # used.
    examples = {}
    for feature in features:
      if feature.type == data_spec_pb2.ColumnType.NUMERICAL:
        examples[feature.name] = tf.constant([1.0, 2.0])
      elif feature.type == data_spec_pb2.ColumnType.CATEGORICAL:
        if inspector.dataspec.columns[
            feature.col_idx].categorical.is_already_integerized:
          examples[feature.name] = tf.constant([1, 2])
        else:
          examples[feature.name] = tf.constant(["a", "b"])
      elif feature.type == data_spec_pb2.ColumnType.CATEGORICAL_SET:
        if inspector.dataspec.columns[
            feature.col_idx].categorical.is_already_integerized:
          examples[feature.name] = tf.ragged.constant([[1, 2], [3]],
                                                      dtype=tf.int32)
        else:
          examples[feature.name] = tf.ragged.constant([["a", "b"], ["c"]],
                                                      dtype=tf.string)
      else:
        raise ValueError("Non supported feature type")

    self.predict(tf.data.Dataset.from_tensor_slices(examples).batch(2))


class _TrainerCallBack(tf.keras.callbacks.Callback):
  """Callback that trains the model at the end of the first epoch."""

  def __init__(self, model: CoreModel):
    self._model = model

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
    task: Optional[TaskType] = Task.CLASSIFICATION) -> tf.data.Dataset:
  """Converts a Panda Dataframe into a TF Dataset.

  Details:
    - Ensures columns have uniform types.
    - If "label" is provided, separate it as a second channel in the tf.Dataset
      (as expected by TF-DF).
    - If "task" is provided, ensure the correct dtype of the label. If the task
      a classification and the label a string, integerize the labels. In this
      case, the label values are extracted from the dataset and ordered
      lexicographically. Warning: This logic won't work as expected if the
      training and testing dataset contains different label values. In such
      case, it is preferable to convert the label to integers beforehand while
      making sure the same encoding is used for all the datasets.
    - Returns "tf.data.from_tensor_slices"

  Args:
    dataframe: Pandas dataframe containing a training or evaluation dataset.
    label: Name of the label column.
    task: Target task of the dataset.

  Returns:
    A TensorFlow Dataset.
  """

  dataframe = dataframe.copy(deep=False)

  if task == Task.CLASSIFICATION and label is not None:
    classification_classes = dataframe[label].unique().tolist()
    classification_classes.sort()
    dataframe[label] = dataframe[label].map(classification_classes.index)

  # Make sure tha missing values for string columns are not represented as
  # float(NaN).
  for col in dataframe.columns:
    if dataframe[col].dtype in [str, object]:
      dataframe[col] = dataframe[col].fillna("")

  if label is not None:
    tf_dataset = tf.data.Dataset.from_tensor_slices(
        (dict(dataframe.drop(label, 1)), dataframe[label].values))
  else:
    tf_dataset = tf.data.Dataset.from_tensor_slices(dict(dataframe))

  # The batch size does not impact the training of TF-DF.
  return tf_dataset.batch(64)


def yggdrasil_model_to_keras_model(src_path: str, dst_path: str):
  """Converts an Yggdrasil model into a Keras model."""

  inspector = inspector_lib.make_inspector(src_path)
  objective = inspector.objective()

  model = CoreModel(
      task=objective.task,
      learner="MANUAL",
      ranking_group=objective.group
      if objective.task == inspector_lib.Task.RANKING else None)

  model._set_from_yggdrasil_model(inspector, src_path)  # pylint: disable=protected-access

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


# pylint: enable=g-doc-args
# pylint: enable=g-doc-return-or-yield
