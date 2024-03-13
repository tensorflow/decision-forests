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

r"""Wrapper around each learning algorithm.

This file is generated automatically by running the following commands:
  bazel build -c opt //tensorflow_decision_forests/keras:wrappers
  bazel-bin/tensorflow_decision_forests/keras/wrappers_wrapper_main\
    > tensorflow_decision_forests/keras/wrappers_pre_generated.py

Please don't change this file directly. Instead, changes the source. The
documentation source is contained in the "GetGenericHyperParameterSpecification"
method of each learner e.g. GetGenericHyperParameterSpecification in
learner/gradient_boosted_trees/gradient_boosted_trees.cc contains the
documentation (and meta-data) used to generate this file.
"""

from typing import Optional, List, Set
import tensorflow as tf
import tf_keras

from tensorflow_decision_forests.keras import core
from tensorflow_decision_forests.component.tuner import tuner as tuner_lib
from yggdrasil_decision_forests.model import abstract_model_pb2  # pylint: disable=unused-import
from yggdrasil_decision_forests.learner import abstract_learner_pb2

TaskType = "abstract_model_pb2.Task"  # pylint: disable=invalid-name
AdvancedArguments = core.AdvancedArguments
MultiTaskItem = core.MultiTaskItem


class CartModel(core.CoreModel):
  r"""Cart learning algorithm.

  A CART (Classification and Regression Trees) a decision tree. The non-leaf
  nodes contains conditions (also known as splits) while the leaf nodes contain
  prediction values. The training dataset is divided in two parts. The first is
  used to grow the tree while the second is used to prune the tree.

  Usage example:

  ```python
  import tensorflow_decision_forests as tfdf
  import pandas as pd

  dataset = pd.read_csv("project/dataset.csv")
  tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="my_label")

  model = tfdf.keras.CartModel()
  model.fit(tf_dataset)

  print(model.summary())
  ```

  Hyper-parameter tuning:

  ```python
  import tensorflow_decision_forests as tfdf
  import pandas as pd

  dataset = pd.read_csv("project/dataset.csv")
  tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="my_label")

  tuner = tfdf.tuner.RandomSearch(num_trials=20)

  # Hyper-parameters to optimize.
  tuner.discret("max_depth", [4, 5, 6, 7])

  model = tfdf.keras.CartModel(tuner=tuner)
  model.fit(tf_dataset)

  print(model.summary())
  ```


  Attributes:
    task: Task to solve (e.g. Task.CLASSIFICATION, Task.REGRESSION,
      Task.RANKING, Task.CATEGORICAL_UPLIFT, Task.NUMERICAL_UPLIFT).
    features: Specify the list and semantic of the input features of the model.
      If not specified, all the available features will be used. If specified
      and if `exclude_non_specified_features=True`, only the features in
      `features` will be used by the model. If "preprocessing" is used,
      `features` corresponds to the output of the preprocessing. In this case,
      it is recommended for the preprocessing to return a dictionary of tensors.
    exclude_non_specified_features: If true, only use the features specified in
      `features`.
    preprocessing: Functional keras model or @tf.function to apply on the input
      feature before the model to train. This preprocessing model can consume
      and return tensors, list of tensors or dictionary of tensors. If
      specified, the model only "sees" the output of the preprocessing (and not
      the raw input). Can be used to prepare the features or to stack multiple
      models on top of each other. Unlike preprocessing done in the tf.dataset,
      the operation in "preprocessing" are serialized with the model.
    postprocessing: Like "preprocessing" but applied on the model output.
    training_preprocessing: Functional keras model or `@tf.function` to apply on
      the input feature, labels, and sample_weight before model training.
    ranking_group: Only for `task=Task.RANKING`. Name of a tf.string feature
      that identifies queries in a query/document ranking task. The ranking
      group is not added automatically for the set of features if
      `exclude_non_specified_features=false`.
    uplift_treatment: Only for task=Task.CATEGORICAL_UPLIFT or
      task=Task.NUMERICAL_UPLIFT. Name of an integer feature that identifies the
      treatment in an uplift problem. The value 0 is reserved for the control
      treatment.
    temp_directory: Temporary directory used to store the model Assets after the
      training, and possibly as a work directory during the training. This
      temporary directory is necessary for the model to be exported after
      training e.g. `model.save(path)`. If not specified, `temp_directory` is
      set to a temporary directory using `tempfile.TemporaryDirectory`. This
      directory is deleted when the model python object is garbage-collected.
    verbose: Verbosity mode. 0 = silent, 1 = small details, 2 = full details.
    hyperparameter_template: Override the default value of the hyper-parameters.
      If None (default) the default parameters of the library are used. If set,
      `default_hyperparameter_template` refers to one of the following
      preconfigured hyper-parameter sets. Those sets outperforms the default
      hyper-parameters (either generally or in specific scenarios). You can omit
      the version (e.g. remove "@v5") to use the last version of the template.
      In this case, the hyper-parameter can change in between releases (not
      recommended for training in production).
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
    multitask: If set, train a multi-task model, that is a model with multiple
      outputs trained to predict different labels. If set, the tf.dataset label
      (i.e. the second selement of the dataset) should be a dictionary of
      label_key:label_values. Only one of `multitask` and `task` can be set.
    allow_na_conditions: If true, the tree training evaluates conditions of the
      type `X is NA` i.e. `X is missing`. Default: False.
    categorical_algorithm: How to learn splits on categorical attributes. -
      `CART`: CART algorithm. Find categorical splits of the form "value \\in
      mask". The solution is exact for binary classification, regression and
      ranking. It is approximated for multi-class classification. This is a good
      first algorithm to use. In case of overfitting (very small dataset, large
      dictionary), the "random" algorithm is a good alternative. - `ONE_HOT`:
      One-hot encoding. Find the optimal categorical split of the form
      "attribute == param". This method is similar (but more efficient) than
      converting converting each possible categorical value into a boolean
      feature. This method is available for comparison purpose and generally
      performs worse than other alternatives. - `RANDOM`: Best splits among a
      set of random candidate. Find the a categorical split of the form "value
      \\in mask" using a random search. This solution can be seen as an
      approximation of the CART algorithm. This method is a strong alternative
      to CART. This algorithm is inspired from section "5.1 Categorical
      Variables" of "Random Forest", 2001.
        Default: "CART".
    categorical_set_split_greedy_sampling: For categorical set splits e.g.
      texts. Probability for a categorical value to be a candidate for the
      positive set. The sampling is applied once per node (i.e. not at every
      step of the greedy optimization). Default: 0.1.
    categorical_set_split_max_num_items: For categorical set splits e.g. texts.
      Maximum number of items (prior to the sampling). If more items are
      available, the least frequent items are ignored. Changing this value is
      similar to change the "max_vocab_count" before loading the dataset, with
      the following exception: With `max_vocab_count`, all the remaining items
      are grouped in a special Out-of-vocabulary item. With `max_num_items`,
      this is not the case. Default: -1.
    categorical_set_split_min_item_frequency: For categorical set splits e.g.
      texts. Minimum number of occurrences of an item to be considered.
      Default: 1.
    growing_strategy: How to grow the tree. - `LOCAL`: Each node is split
      independently of the other nodes. In other words, as long as a node
      satisfy the splits "constraints (e.g. maximum depth, minimum number of
      observations), the node will be split. This is the "classical" way to grow
      decision trees. - `BEST_FIRST_GLOBAL`: The node with the best loss
      reduction among all the nodes of the tree is selected for splitting. This
      method is also called "best first" or "leaf-wise growth". See "Best-first
      decision tree learning", Shi and "Additive logistic regression : A
      statistical view of boosting", Friedman for more details. Default:
      "LOCAL".
    honest: In honest trees, different training examples are used to infer the
      structure and the leaf values. This regularization technique trades
      examples for bias estimates. It might increase or reduce the quality of
      the model. See "Generalized Random Forests", Athey et al. In this paper,
      Honest trees are trained with the Random Forest algorithm with a sampling
      without replacement. Default: False.
    honest_fixed_separation: For honest trees only i.e. honest=true. If true, a
      new random separation is generated for each tree. If false, the same
      separation is used for all the trees (e.g., in Gradient Boosted Trees
      containing multiple trees). Default: False.
    honest_ratio_leaf_examples: For honest trees only i.e. honest=true. Ratio of
      examples used to set the leaf values. Default: 0.5.
    in_split_min_examples_check: Whether to check the `min_examples` constraint
      in the split search (i.e. splits leading to one child having less than
      `min_examples` examples are considered invalid) or before the split search
      (i.e. a node can be derived only if it contains more than `min_examples`
      examples). If false, there can be nodes with less than `min_examples`
      training examples. Default: True.
    keep_non_leaf_label_distribution: Whether to keep the node value (i.e. the
      distribution of the labels of the training examples) of non-leaf nodes.
      This information is not used during serving, however it can be used for
      model interpretation as well as hyper parameter tuning. This can take lots
      of space, sometimes accounting for half of the model size. Default: True.
    max_depth: Maximum depth of the tree. `max_depth=1` means that all trees
      will be roots. `max_depth=-1` means that tree depth is not restricted by
      this parameter. Values <= -2 will be ignored. Default: 16.
    max_num_nodes: Maximum number of nodes in the tree. Set to -1 to disable
      this limit. Only available for `growing_strategy=BEST_FIRST_GLOBAL`.
      Default: None.
    maximum_model_size_in_memory_in_bytes: Limit the size of the model when
      stored in ram. Different algorithms can enforce this limit differently.
      Note that when models are compiled into an inference, the size of the
      inference engine is generally much smaller than the original model.
      Default: -1.0.
    maximum_training_duration_seconds: Maximum training duration of the model
      expressed in seconds. Each learning algorithm is free to use this
      parameter at it sees fit. Enabling maximum training duration makes the
      model training non-deterministic. Default: -1.0.
    min_examples: Minimum number of examples in a node. Default: 5.
    missing_value_policy: Method used to handle missing attribute values. -
      `GLOBAL_IMPUTATION`: Missing attribute values are imputed, with the mean
      (in case of numerical attribute) or the most-frequent-item (in case of
      categorical attribute) computed on the entire dataset (i.e. the
      information contained in the data spec). - `LOCAL_IMPUTATION`: Missing
      attribute values are imputed with the mean (numerical attribute) or
      most-frequent-item (in the case of categorical attribute) evaluated on the
      training examples in the current node. - `RANDOM_LOCAL_IMPUTATION`:
      Missing attribute values are imputed from randomly sampled values from the
      training examples in the current node. This method was proposed by Clinic
      et al. in "Random Survival Forests"
      (https://projecteuclid.org/download/pdfview_1/euclid.aoas/1223908043).
        Default: "GLOBAL_IMPUTATION".
    num_candidate_attributes: Number of unique valid attributes tested for each
      node. An attribute is valid if it has at least a valid split. If
      `num_candidate_attributes=0`, the value is set to the classical default
      value for Random Forest: `sqrt(number of input attributes)` in case of
      classification and `number_of_input_attributes / 3` in case of regression.
      If `num_candidate_attributes=-1`, all the attributes are tested. Default:
      0.
    num_candidate_attributes_ratio: Ratio of attributes tested at each node. If
      set, it is equivalent to `num_candidate_attributes =
      number_of_input_features x num_candidate_attributes_ratio`. The possible
      values are between ]0, and 1] as well as -1. If not set or equal to -1,
      the `num_candidate_attributes` is used. Default: -1.0.
    pure_serving_model: Clear the model from any information that is not
      required for model serving. This includes debugging, model interpretation
      and other meta-data. The size of the serialized model can be reduced
      significatively (50% model size reduction is common). This parameter has
      no impact on the quality, serving speed or RAM usage of model serving.
      Default: False.
    random_seed: Random seed for the training of the model. Learners are
      expected to be deterministic by the random seed. Default: 123456.
    sorting_strategy: How are sorted the numerical features in order to find the
      splits - PRESORT: The features are pre-sorted at the start of the
      training. This solution is faster but consumes much more memory than
      IN_NODE. - IN_NODE: The features are sorted just before being used in the
      node. This solution is slow but consumes little amount of memory. .
      Default: "PRESORT".
    sparse_oblique_max_num_projections: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Maximum number of projections (applied after
      the num_projections_exponent). Oblique splits try out
      max(p^num_projections_exponent, max_num_projections) random projections
      for choosing a split, where p is the number of numerical features.
      Increasing "max_num_projections" increases the training time but not the
      inference time. In late stage model development, if every bit of accuracy
      if important, increase this value. The paper "Sparse Projection Oblique
      Random Forests" (Tomita et al, 2020) does not define this hyperparameter.
      Default: None.
    sparse_oblique_normalization: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Normalization applied on the features, before
      applying the sparse oblique projections. - `NONE`: No normalization. -
      `STANDARD_DEVIATION`: Normalize the feature by the estimated standard
      deviation on the entire train dataset. Also known as Z-Score
      normalization. - `MIN_MAX`: Normalize the feature by the range (i.e.
      max-min) estimated on the entire train dataset. Default: None.
    sparse_oblique_num_projections_exponent: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Controls of the number of random projections
      to test at each node. Increasing this value very likely improves the
      quality of the model, drastically increases the training time, and doe not
      impact the inference time. Oblique splits try out
      max(p^num_projections_exponent, max_num_projections) random projections
      for choosing a split, where p is the number of numerical features.
      Therefore, increasing this `num_projections_exponent` and possibly
      `max_num_projections` may improve model quality, but will also
      significantly increase training time. Note that the complexity of
      (classic) Random Forests is roughly proportional to
      `num_projections_exponent=0.5`, since it considers sqrt(num_features) for
      a split. The complexity of (classic) GBDT is roughly proportional to
      `num_projections_exponent=1`, since it considers all features for a split.
      The paper "Sparse Projection Oblique Random Forests" (Tomita et al, 2020)
      recommends values in [1/4, 2]. Default: None.
    sparse_oblique_projection_density_factor: Density of the projections as an
      exponent of the number of features. Independently for each projection,
      each feature has a probability "projection_density_factor / num_features"
      to be considered in the projection. The paper "Sparse Projection Oblique
      Random Forests" (Tomita et al, 2020) calls this parameter `lambda` and
      recommends values in [1, 5]. Increasing this value increases training and
      inference time (on average). This value is best tuned for each dataset.
      Default: None.
    sparse_oblique_weights: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Possible values: - `BINARY`: The oblique
      weights are sampled in {-1,1} (default). - `CONTINUOUS`: The oblique
      weights are be sampled in [-1,1]. Default: None.
    split_axis: What structure of split to consider for numerical features. -
      `AXIS_ALIGNED`: Axis aligned splits (i.e. one condition at a time). This
      is the "classical" way to train a tree. Default value. - `SPARSE_OBLIQUE`:
      Sparse oblique splits (i.e. splits one a small number of features) from
      "Sparse Projection Oblique Random Forests", Tomita et al., 2020. Default:
      "AXIS_ALIGNED".
    uplift_min_examples_in_treatment: For uplift models only. Minimum number of
      examples per treatment in a node. Default: 5.
    uplift_split_score: For uplift models only. Splitter score i.e. score
      optimized by the splitters. The scores are introduced in "Decision trees
      for uplift modeling with single and multiple treatments", Rzepakowski et
      al. Notation: `p` probability / average value of the positive outcome, `q`
      probability / average value in the control group. - `KULLBACK_LEIBLER` or
      `KL`: - p log (p/q) - `EUCLIDEAN_DISTANCE` or `ED`: (p-q)^2 -
      `CHI_SQUARED` or `CS`: (p-q)^2/q
        Default: "KULLBACK_LEIBLER".
    validation_ratio: Ratio of the training dataset used to create the
      validation dataset for pruning the tree. If set to 0, the entire dataset
      is used for training, and the tree is not pruned. Default: 0.1.
  """

  @core._list_explicit_arguments
  def __init__(
      self,
      task: Optional[TaskType] = core.Task.CLASSIFICATION,
      features: Optional[List[core.FeatureUsage]] = None,
      exclude_non_specified_features: Optional[bool] = False,
      preprocessing: Optional["tf_keras.models.Functional"] = None,
      postprocessing: Optional["tf_keras.models.Functional"] = None,
      training_preprocessing: Optional["tf_keras.models.Functional"] = None,
      ranking_group: Optional[str] = None,
      uplift_treatment: Optional[str] = None,
      temp_directory: Optional[str] = None,
      verbose: int = 1,
      hyperparameter_template: Optional[str] = None,
      advanced_arguments: Optional[AdvancedArguments] = None,
      num_threads: Optional[int] = None,
      name: Optional[str] = None,
      max_vocab_count: Optional[int] = 2000,
      try_resume_training: Optional[bool] = True,
      check_dataset: Optional[bool] = True,
      tuner: Optional[tuner_lib.Tuner] = None,
      discretize_numerical_features: bool = False,
      num_discretized_numerical_bins: int = 255,
      multitask: Optional[List[MultiTaskItem]] = None,
      allow_na_conditions: Optional[bool] = False,
      categorical_algorithm: Optional[str] = "CART",
      categorical_set_split_greedy_sampling: Optional[float] = 0.1,
      categorical_set_split_max_num_items: Optional[int] = -1,
      categorical_set_split_min_item_frequency: Optional[int] = 1,
      growing_strategy: Optional[str] = "LOCAL",
      honest: Optional[bool] = False,
      honest_fixed_separation: Optional[bool] = False,
      honest_ratio_leaf_examples: Optional[float] = 0.5,
      in_split_min_examples_check: Optional[bool] = True,
      keep_non_leaf_label_distribution: Optional[bool] = True,
      max_depth: Optional[int] = 16,
      max_num_nodes: Optional[int] = None,
      maximum_model_size_in_memory_in_bytes: Optional[float] = -1.0,
      maximum_training_duration_seconds: Optional[float] = -1.0,
      min_examples: Optional[int] = 5,
      missing_value_policy: Optional[str] = "GLOBAL_IMPUTATION",
      num_candidate_attributes: Optional[int] = 0,
      num_candidate_attributes_ratio: Optional[float] = -1.0,
      pure_serving_model: Optional[bool] = False,
      random_seed: Optional[int] = 123456,
      sorting_strategy: Optional[str] = "PRESORT",
      sparse_oblique_max_num_projections: Optional[int] = None,
      sparse_oblique_normalization: Optional[str] = None,
      sparse_oblique_num_projections_exponent: Optional[float] = None,
      sparse_oblique_projection_density_factor: Optional[float] = None,
      sparse_oblique_weights: Optional[str] = None,
      split_axis: Optional[str] = "AXIS_ALIGNED",
      uplift_min_examples_in_treatment: Optional[int] = 5,
      uplift_split_score: Optional[str] = "KULLBACK_LEIBLER",
      validation_ratio: Optional[float] = 0.1,
      explicit_args: Optional[Set[str]] = None,
  ):

    learner_params = {
        "allow_na_conditions": allow_na_conditions,
        "categorical_algorithm": categorical_algorithm,
        "categorical_set_split_greedy_sampling": (
            categorical_set_split_greedy_sampling
        ),
        "categorical_set_split_max_num_items": (
            categorical_set_split_max_num_items
        ),
        "categorical_set_split_min_item_frequency": (
            categorical_set_split_min_item_frequency
        ),
        "growing_strategy": growing_strategy,
        "honest": honest,
        "honest_fixed_separation": honest_fixed_separation,
        "honest_ratio_leaf_examples": honest_ratio_leaf_examples,
        "in_split_min_examples_check": in_split_min_examples_check,
        "keep_non_leaf_label_distribution": keep_non_leaf_label_distribution,
        "max_depth": max_depth,
        "max_num_nodes": max_num_nodes,
        "maximum_model_size_in_memory_in_bytes": (
            maximum_model_size_in_memory_in_bytes
        ),
        "maximum_training_duration_seconds": maximum_training_duration_seconds,
        "min_examples": min_examples,
        "missing_value_policy": missing_value_policy,
        "num_candidate_attributes": num_candidate_attributes,
        "num_candidate_attributes_ratio": num_candidate_attributes_ratio,
        "pure_serving_model": pure_serving_model,
        "random_seed": random_seed,
        "sorting_strategy": sorting_strategy,
        "sparse_oblique_max_num_projections": (
            sparse_oblique_max_num_projections
        ),
        "sparse_oblique_normalization": sparse_oblique_normalization,
        "sparse_oblique_num_projections_exponent": (
            sparse_oblique_num_projections_exponent
        ),
        "sparse_oblique_projection_density_factor": (
            sparse_oblique_projection_density_factor
        ),
        "sparse_oblique_weights": sparse_oblique_weights,
        "split_axis": split_axis,
        "uplift_min_examples_in_treatment": uplift_min_examples_in_treatment,
        "uplift_split_score": uplift_split_score,
        "validation_ratio": validation_ratio,
    }

    if hyperparameter_template is not None:
      learner_params = core._apply_hp_template(
          learner_params,
          hyperparameter_template,
          self.predefined_hyperparameters(),
          explicit_args,
      )

    super(CartModel, self).__init__(
        task=task,
        learner="CART",
        learner_params=learner_params,
        features=features,
        exclude_non_specified_features=exclude_non_specified_features,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        training_preprocessing=training_preprocessing,
        ranking_group=ranking_group,
        uplift_treatment=uplift_treatment,
        temp_directory=temp_directory,
        verbose=verbose,
        advanced_arguments=advanced_arguments,
        num_threads=num_threads,
        name=name,
        max_vocab_count=max_vocab_count,
        try_resume_training=try_resume_training,
        check_dataset=check_dataset,
        tuner=tuner,
        discretize_numerical_features=discretize_numerical_features,
        num_discretized_numerical_bins=num_discretized_numerical_bins,
        multitask=multitask,
    )

  @staticmethod
  def predefined_hyperparameters() -> List[core.HyperParameterTemplate]:
    return []

  @staticmethod
  def capabilities() -> abstract_learner_pb2.LearnerCapabilities:
    return abstract_learner_pb2.LearnerCapabilities(
        support_partial_cache_dataset_format=False
    )

class DistributedGradientBoostedTreesModel(core.CoreModel):
  r"""Distributed Gradient Boosted Trees learning algorithm.

  Exact distributed version of the Gradient Boosted Tree learning algorithm. See
  the documentation of the non-distributed Gradient Boosted Tree learning
  algorithm for an introduction to GBTs.

  Usage example:

  ```python
  import tensorflow_decision_forests as tfdf
  import pandas as pd

  dataset = pd.read_csv("project/dataset.csv")
  tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="my_label")

  model = tfdf.keras.DistributedGradientBoostedTreesModel()
  model.fit(tf_dataset)

  print(model.summary())
  ```

  Hyper-parameter tuning:

  ```python
  import tensorflow_decision_forests as tfdf
  import pandas as pd

  dataset = pd.read_csv("project/dataset.csv")
  tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="my_label")

  tuner = tfdf.tuner.RandomSearch(num_trials=20)

  # Hyper-parameters to optimize.
  tuner.discret("max_depth", [4, 5, 6, 7])

  model = tfdf.keras.DistributedGradientBoostedTreesModel(tuner=tuner)
  model.fit(tf_dataset)

  print(model.summary())
  ```


  Attributes:
    task: Task to solve (e.g. Task.CLASSIFICATION, Task.REGRESSION,
      Task.RANKING, Task.CATEGORICAL_UPLIFT, Task.NUMERICAL_UPLIFT).
    features: Specify the list and semantic of the input features of the model.
      If not specified, all the available features will be used. If specified
      and if `exclude_non_specified_features=True`, only the features in
      `features` will be used by the model. If "preprocessing" is used,
      `features` corresponds to the output of the preprocessing. In this case,
      it is recommended for the preprocessing to return a dictionary of tensors.
    exclude_non_specified_features: If true, only use the features specified in
      `features`.
    preprocessing: Functional keras model or @tf.function to apply on the input
      feature before the model to train. This preprocessing model can consume
      and return tensors, list of tensors or dictionary of tensors. If
      specified, the model only "sees" the output of the preprocessing (and not
      the raw input). Can be used to prepare the features or to stack multiple
      models on top of each other. Unlike preprocessing done in the tf.dataset,
      the operation in "preprocessing" are serialized with the model.
    postprocessing: Like "preprocessing" but applied on the model output.
    training_preprocessing: Functional keras model or `@tf.function` to apply on
      the input feature, labels, and sample_weight before model training.
    ranking_group: Only for `task=Task.RANKING`. Name of a tf.string feature
      that identifies queries in a query/document ranking task. The ranking
      group is not added automatically for the set of features if
      `exclude_non_specified_features=false`.
    uplift_treatment: Only for task=Task.CATEGORICAL_UPLIFT or
      task=Task.NUMERICAL_UPLIFT. Name of an integer feature that identifies the
      treatment in an uplift problem. The value 0 is reserved for the control
      treatment.
    temp_directory: Temporary directory used to store the model Assets after the
      training, and possibly as a work directory during the training. This
      temporary directory is necessary for the model to be exported after
      training e.g. `model.save(path)`. If not specified, `temp_directory` is
      set to a temporary directory using `tempfile.TemporaryDirectory`. This
      directory is deleted when the model python object is garbage-collected.
    verbose: Verbosity mode. 0 = silent, 1 = small details, 2 = full details.
    hyperparameter_template: Override the default value of the hyper-parameters.
      If None (default) the default parameters of the library are used. If set,
      `default_hyperparameter_template` refers to one of the following
      preconfigured hyper-parameter sets. Those sets outperforms the default
      hyper-parameters (either generally or in specific scenarios). You can omit
      the version (e.g. remove "@v5") to use the last version of the template.
      In this case, the hyper-parameter can change in between releases (not
      recommended for training in production).
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
    multitask: If set, train a multi-task model, that is a model with multiple
      outputs trained to predict different labels. If set, the tf.dataset label
      (i.e. the second selement of the dataset) should be a dictionary of
      label_key:label_values. Only one of `multitask` and `task` can be set.
    apply_link_function: If true, applies the link function (a.k.a. activation
      function), if any, before returning the model prediction. If false,
      returns the pre-link function model output. For example, in the case of
      binary classification, the pre-link function output is a logic while the
      post-link function is a probability. Default: True.
    force_numerical_discretization: If false, only the numerical column
      safisfying "max_unique_values_for_discretized_numerical" will be
      discretized. If true, all the numerical columns will be discretized.
      Columns with more than "max_unique_values_for_discretized_numerical"
      unique values will be approximated with
      "max_unique_values_for_discretized_numerical" bins. This parameter will
      impact the model training. Default: False.
    max_depth: Maximum depth of the tree. `max_depth=1` means that all trees
      will be roots. `max_depth=-1` means that tree depth is not restricted by
      this parameter. Values <= -2 will be ignored. Default: 6.
    max_unique_values_for_discretized_numerical: Maximum number of unique value
      of a numerical feature to allow its pre-discretization. In case of large
      datasets, discretized numerical features with a small number of unique
      values are more efficient to learn than classical / non-discretized
      numerical features. This parameter does not impact the final model.
      However, it can speed-up or slown the training. Default: 16000.
    maximum_model_size_in_memory_in_bytes: Limit the size of the model when
      stored in ram. Different algorithms can enforce this limit differently.
      Note that when models are compiled into an inference, the size of the
      inference engine is generally much smaller than the original model.
      Default: -1.0.
    maximum_training_duration_seconds: Maximum training duration of the model
      expressed in seconds. Each learning algorithm is free to use this
      parameter at it sees fit. Enabling maximum training duration makes the
      model training non-deterministic. Default: -1.0.
    min_examples: Minimum number of examples in a node. Default: 5.
    num_candidate_attributes: Number of unique valid attributes tested for each
      node. An attribute is valid if it has at least a valid split. If
      `num_candidate_attributes=0`, the value is set to the classical default
      value for Random Forest: `sqrt(number of input attributes)` in case of
      classification and `number_of_input_attributes / 3` in case of regression.
      If `num_candidate_attributes=-1`, all the attributes are tested. Default:
      -1.
    num_candidate_attributes_ratio: Ratio of attributes tested at each node. If
      set, it is equivalent to `num_candidate_attributes =
      number_of_input_features x num_candidate_attributes_ratio`. The possible
      values are between ]0, and 1] as well as -1. If not set or equal to -1,
      the `num_candidate_attributes` is used. Default: -1.0.
    num_trees: Maximum number of decision trees. The effective number of trained
      tree can be smaller if early stopping is enabled. Default: 300.
    pure_serving_model: Clear the model from any information that is not
      required for model serving. This includes debugging, model interpretation
      and other meta-data. The size of the serialized model can be reduced
      significatively (50% model size reduction is common). This parameter has
      no impact on the quality, serving speed or RAM usage of model serving.
      Default: False.
    random_seed: Random seed for the training of the model. Learners are
      expected to be deterministic by the random seed. Default: 123456.
    shrinkage: Coefficient applied to each tree prediction. A small value (0.02)
      tends to give more accurate results (assuming enough trees are trained),
      but results in larger models. Analogous to neural network learning rate.
      Default: 0.1.
    use_hessian_gain: Use true, uses a formulation of split gain with a hessian
      term i.e. optimizes the splits to minimize the variance of "gradient /
      hessian. Available for all losses except regression. Default: False.
    worker_logs: If true, workers will print training logs. Default: True.
  """

  @core._list_explicit_arguments
  def __init__(
      self,
      task: Optional[TaskType] = core.Task.CLASSIFICATION,
      features: Optional[List[core.FeatureUsage]] = None,
      exclude_non_specified_features: Optional[bool] = False,
      preprocessing: Optional["tf_keras.models.Functional"] = None,
      postprocessing: Optional["tf_keras.models.Functional"] = None,
      training_preprocessing: Optional["tf_keras.models.Functional"] = None,
      ranking_group: Optional[str] = None,
      uplift_treatment: Optional[str] = None,
      temp_directory: Optional[str] = None,
      verbose: int = 1,
      hyperparameter_template: Optional[str] = None,
      advanced_arguments: Optional[AdvancedArguments] = None,
      num_threads: Optional[int] = None,
      name: Optional[str] = None,
      max_vocab_count: Optional[int] = 2000,
      try_resume_training: Optional[bool] = True,
      check_dataset: Optional[bool] = True,
      tuner: Optional[tuner_lib.Tuner] = None,
      discretize_numerical_features: bool = False,
      num_discretized_numerical_bins: int = 255,
      multitask: Optional[List[MultiTaskItem]] = None,
      apply_link_function: Optional[bool] = True,
      force_numerical_discretization: Optional[bool] = False,
      max_depth: Optional[int] = 6,
      max_unique_values_for_discretized_numerical: Optional[int] = 16000,
      maximum_model_size_in_memory_in_bytes: Optional[float] = -1.0,
      maximum_training_duration_seconds: Optional[float] = -1.0,
      min_examples: Optional[int] = 5,
      num_candidate_attributes: Optional[int] = -1,
      num_candidate_attributes_ratio: Optional[float] = -1.0,
      num_trees: Optional[int] = 300,
      pure_serving_model: Optional[bool] = False,
      random_seed: Optional[int] = 123456,
      shrinkage: Optional[float] = 0.1,
      use_hessian_gain: Optional[bool] = False,
      worker_logs: Optional[bool] = True,
      explicit_args: Optional[Set[str]] = None,
  ):

    learner_params = {
        "apply_link_function": apply_link_function,
        "force_numerical_discretization": force_numerical_discretization,
        "max_depth": max_depth,
        "max_unique_values_for_discretized_numerical": (
            max_unique_values_for_discretized_numerical
        ),
        "maximum_model_size_in_memory_in_bytes": (
            maximum_model_size_in_memory_in_bytes
        ),
        "maximum_training_duration_seconds": maximum_training_duration_seconds,
        "min_examples": min_examples,
        "num_candidate_attributes": num_candidate_attributes,
        "num_candidate_attributes_ratio": num_candidate_attributes_ratio,
        "num_trees": num_trees,
        "pure_serving_model": pure_serving_model,
        "random_seed": random_seed,
        "shrinkage": shrinkage,
        "use_hessian_gain": use_hessian_gain,
        "worker_logs": worker_logs,
    }

    if hyperparameter_template is not None:
      learner_params = core._apply_hp_template(
          learner_params,
          hyperparameter_template,
          self.predefined_hyperparameters(),
          explicit_args,
      )

    super(DistributedGradientBoostedTreesModel, self).__init__(
        task=task,
        learner="DISTRIBUTED_GRADIENT_BOOSTED_TREES",
        learner_params=learner_params,
        features=features,
        exclude_non_specified_features=exclude_non_specified_features,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        training_preprocessing=training_preprocessing,
        ranking_group=ranking_group,
        uplift_treatment=uplift_treatment,
        temp_directory=temp_directory,
        verbose=verbose,
        advanced_arguments=advanced_arguments,
        num_threads=num_threads,
        name=name,
        max_vocab_count=max_vocab_count,
        try_resume_training=try_resume_training,
        check_dataset=check_dataset,
        tuner=tuner,
        discretize_numerical_features=discretize_numerical_features,
        num_discretized_numerical_bins=num_discretized_numerical_bins,
        multitask=multitask,
    )

  @staticmethod
  def predefined_hyperparameters() -> List[core.HyperParameterTemplate]:
    return []

  @staticmethod
  def capabilities() -> abstract_learner_pb2.LearnerCapabilities:
    return abstract_learner_pb2.LearnerCapabilities(
        support_partial_cache_dataset_format=True
    )

class GradientBoostedTreesModel(core.CoreModel):
  r"""Gradient Boosted Trees learning algorithm.

  A [Gradient Boosted Trees](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)
  (GBT), also known as Gradient Boosted Decision Trees (GBDT) or Gradient
  Boosted Machines (GBM),  is a set of shallow decision trees trained
  sequentially. Each tree is trained to predict and then "correct" for the
  errors of the previously trained trees (more precisely each tree predict the
  gradient of the loss relative to the model output).

  Usage example:

  ```python
  import tensorflow_decision_forests as tfdf
  import pandas as pd

  dataset = pd.read_csv("project/dataset.csv")
  tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="my_label")

  model = tfdf.keras.GradientBoostedTreesModel()
  model.fit(tf_dataset)

  print(model.summary())
  ```

  Hyper-parameter tuning:

  ```python
  import tensorflow_decision_forests as tfdf
  import pandas as pd

  dataset = pd.read_csv("project/dataset.csv")
  tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="my_label")

  tuner = tfdf.tuner.RandomSearch(num_trials=20)

  # Hyper-parameters to optimize.
  tuner.discret("max_depth", [4, 5, 6, 7])

  model = tfdf.keras.GradientBoostedTreesModel(tuner=tuner)
  model.fit(tf_dataset)

  print(model.summary())
  ```


  Attributes:
    task: Task to solve (e.g. Task.CLASSIFICATION, Task.REGRESSION,
      Task.RANKING, Task.CATEGORICAL_UPLIFT, Task.NUMERICAL_UPLIFT).
    features: Specify the list and semantic of the input features of the model.
      If not specified, all the available features will be used. If specified
      and if `exclude_non_specified_features=True`, only the features in
      `features` will be used by the model. If "preprocessing" is used,
      `features` corresponds to the output of the preprocessing. In this case,
      it is recommended for the preprocessing to return a dictionary of tensors.
    exclude_non_specified_features: If true, only use the features specified in
      `features`.
    preprocessing: Functional keras model or @tf.function to apply on the input
      feature before the model to train. This preprocessing model can consume
      and return tensors, list of tensors or dictionary of tensors. If
      specified, the model only "sees" the output of the preprocessing (and not
      the raw input). Can be used to prepare the features or to stack multiple
      models on top of each other. Unlike preprocessing done in the tf.dataset,
      the operation in "preprocessing" are serialized with the model.
    postprocessing: Like "preprocessing" but applied on the model output.
    training_preprocessing: Functional keras model or `@tf.function` to apply on
      the input feature, labels, and sample_weight before model training.
    ranking_group: Only for `task=Task.RANKING`. Name of a tf.string feature
      that identifies queries in a query/document ranking task. The ranking
      group is not added automatically for the set of features if
      `exclude_non_specified_features=false`.
    uplift_treatment: Only for task=Task.CATEGORICAL_UPLIFT or
      task=Task.NUMERICAL_UPLIFT. Name of an integer feature that identifies the
      treatment in an uplift problem. The value 0 is reserved for the control
      treatment.
    temp_directory: Temporary directory used to store the model Assets after the
      training, and possibly as a work directory during the training. This
      temporary directory is necessary for the model to be exported after
      training e.g. `model.save(path)`. If not specified, `temp_directory` is
      set to a temporary directory using `tempfile.TemporaryDirectory`. This
      directory is deleted when the model python object is garbage-collected.
    verbose: Verbosity mode. 0 = silent, 1 = small details, 2 = full details.
    hyperparameter_template: Override the default value of the hyper-parameters.
      If None (default) the default parameters of the library are used. If set,
      `default_hyperparameter_template` refers to one of the following
      preconfigured hyper-parameter sets. Those sets outperforms the default
      hyper-parameters (either generally or in specific scenarios). You can omit
      the version (e.g. remove "@v5") to use the last version of the template.
      In this case, the hyper-parameter can change in between releases (not
      recommended for training in production). - better_default@v1: A
      configuration that is generally better than the default parameters without
      being more expensive. The parameters are:
      growing_strategy="BEST_FIRST_GLOBAL". - benchmark_rank1@v1: Top ranking
      hyper-parameters on our benchmark slightly modified to run in reasonable
      time. The parameters are: growing_strategy="BEST_FIRST_GLOBAL",
      categorical_algorithm="RANDOM", split_axis="SPARSE_OBLIQUE",
      sparse_oblique_normalization="MIN_MAX",
      sparse_oblique_num_projections_exponent=1.0.
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
    multitask: If set, train a multi-task model, that is a model with multiple
      outputs trained to predict different labels. If set, the tf.dataset label
      (i.e. the second selement of the dataset) should be a dictionary of
      label_key:label_values. Only one of `multitask` and `task` can be set.
    adapt_subsample_for_maximum_training_duration: Control how the maximum
      training duration (if set) is applied. If false, the training stop when
      the time is used. If true, the size of the sampled datasets used train
      individual trees are adapted dynamically so that all the trees are trained
      in time. Default: False.
    allow_na_conditions: If true, the tree training evaluates conditions of the
      type `X is NA` i.e. `X is missing`. Default: False.
    apply_link_function: If true, applies the link function (a.k.a. activation
      function), if any, before returning the model prediction. If false,
      returns the pre-link function model output. For example, in the case of
      binary classification, the pre-link function output is a logic while the
      post-link function is a probability. Default: True.
    categorical_algorithm: How to learn splits on categorical attributes. -
      `CART`: CART algorithm. Find categorical splits of the form "value \\in
      mask". The solution is exact for binary classification, regression and
      ranking. It is approximated for multi-class classification. This is a good
      first algorithm to use. In case of overfitting (very small dataset, large
      dictionary), the "random" algorithm is a good alternative. - `ONE_HOT`:
      One-hot encoding. Find the optimal categorical split of the form
      "attribute == param". This method is similar (but more efficient) than
      converting converting each possible categorical value into a boolean
      feature. This method is available for comparison purpose and generally
      performs worse than other alternatives. - `RANDOM`: Best splits among a
      set of random candidate. Find the a categorical split of the form "value
      \\in mask" using a random search. This solution can be seen as an
      approximation of the CART algorithm. This method is a strong alternative
      to CART. This algorithm is inspired from section "5.1 Categorical
      Variables" of "Random Forest", 2001.
        Default: "CART".
    categorical_set_split_greedy_sampling: For categorical set splits e.g.
      texts. Probability for a categorical value to be a candidate for the
      positive set. The sampling is applied once per node (i.e. not at every
      step of the greedy optimization). Default: 0.1.
    categorical_set_split_max_num_items: For categorical set splits e.g. texts.
      Maximum number of items (prior to the sampling). If more items are
      available, the least frequent items are ignored. Changing this value is
      similar to change the "max_vocab_count" before loading the dataset, with
      the following exception: With `max_vocab_count`, all the remaining items
      are grouped in a special Out-of-vocabulary item. With `max_num_items`,
      this is not the case. Default: -1.
    categorical_set_split_min_item_frequency: For categorical set splits e.g.
      texts. Minimum number of occurrences of an item to be considered.
      Default: 1.
    compute_permutation_variable_importance: If true, compute the permutation
      variable importance of the model at the end of the training using the
      validation dataset. Enabling this feature can increase the training time
      significantly. Default: False.
    dart_dropout: Dropout rate applied when using the DART i.e. when
      forest_extraction=DART. Default: 0.01.
    early_stopping: Early stopping detects the overfitting of the model and
      halts it training using the validation dataset. If not provided directly,
      the validation dataset is extracted from the training dataset (see
      "validation_ratio" parameter): - `NONE`: No early stopping. All the
      num_trees are trained and kept. - `MIN_LOSS_FINAL`: All the num_trees are
      trained. The model is then truncated to minimize the validation loss i.e.
      some of the trees are discarded as to minimum the validation loss. -
      `LOSS_INCREASE`: Classical early stopping. Stop the training when the
      validation does not decrease for `early_stopping_num_trees_look_ahead`
      trees. Default: "LOSS_INCREASE".
    early_stopping_initial_iteration: 0-based index of the first iteration
      considered for early stopping computation. Increasing this value prevents
      too early stopping due to noisy initial iterations of the learner.
      Default: 10.
    early_stopping_num_trees_look_ahead: Rolling number of trees used to detect
      validation loss increase and trigger early stopping. Default: 30.
    focal_loss_alpha: EXPERIMENTAL. Weighting parameter for focal loss, positive
      samples weighted by alpha, negative samples by (1-alpha). The default 0.5
      value means no active class-level weighting. Only used with focal loss
      i.e. `loss="BINARY_FOCAL_LOSS"` Default: 0.5.
    focal_loss_gamma: EXPERIMENTAL. Exponent of the misprediction exponent term
      in focal loss, corresponds to gamma parameter in
      https://arxiv.org/pdf/1708.02002.pdf. Only used with focal loss i.e.
        `loss="BINARY_FOCAL_LOSS"` Default: 2.0.
    forest_extraction: How to construct the forest: - MART: For Multiple
      Additive Regression Trees. The "classical" way to build a GBDT i.e. each
      tree tries to "correct" the mistakes of the previous trees. - DART: For
      Dropout Additive Regression Trees. A modification of MART proposed in
      http://proceedings.mlr.press/v38/korlakaivinayak15.pdf. Here, each tree
        tries to "correct" the mistakes of a random subset of the previous
        trees.
      Default: "MART".
    goss_alpha: Alpha parameter for the GOSS (Gradient-based One-Side Sampling;
      "See LightGBM: A Highly Efficient Gradient Boosting Decision Tree")
      sampling method. Default: 0.2.
    goss_beta: Beta parameter for the GOSS (Gradient-based One-Side Sampling)
      sampling method. Default: 0.1.
    growing_strategy: How to grow the tree. - `LOCAL`: Each node is split
      independently of the other nodes. In other words, as long as a node
      satisfy the splits "constraints (e.g. maximum depth, minimum number of
      observations), the node will be split. This is the "classical" way to grow
      decision trees. - `BEST_FIRST_GLOBAL`: The node with the best loss
      reduction among all the nodes of the tree is selected for splitting. This
      method is also called "best first" or "leaf-wise growth". See "Best-first
      decision tree learning", Shi and "Additive logistic regression : A
      statistical view of boosting", Friedman for more details. Default:
      "LOCAL".
    honest: In honest trees, different training examples are used to infer the
      structure and the leaf values. This regularization technique trades
      examples for bias estimates. It might increase or reduce the quality of
      the model. See "Generalized Random Forests", Athey et al. In this paper,
      Honest trees are trained with the Random Forest algorithm with a sampling
      without replacement. Default: False.
    honest_fixed_separation: For honest trees only i.e. honest=true. If true, a
      new random separation is generated for each tree. If false, the same
      separation is used for all the trees (e.g., in Gradient Boosted Trees
      containing multiple trees). Default: False.
    honest_ratio_leaf_examples: For honest trees only i.e. honest=true. Ratio of
      examples used to set the leaf values. Default: 0.5.
    in_split_min_examples_check: Whether to check the `min_examples` constraint
      in the split search (i.e. splits leading to one child having less than
      `min_examples` examples are considered invalid) or before the split search
      (i.e. a node can be derived only if it contains more than `min_examples`
      examples). If false, there can be nodes with less than `min_examples`
      training examples. Default: True.
    keep_non_leaf_label_distribution: Whether to keep the node value (i.e. the
      distribution of the labels of the training examples) of non-leaf nodes.
      This information is not used during serving, however it can be used for
      model interpretation as well as hyper parameter tuning. This can take lots
      of space, sometimes accounting for half of the model size. Default: True.
    l1_regularization: L1 regularization applied to the training loss. Impact
      the tree structures and lead values. Default: 0.0.
    l2_categorical_regularization: L2 regularization applied to the training
      loss for categorical features. Impact the tree structures and lead values.
      Default: 1.0.
    l2_regularization: L2 regularization applied to the training loss for all
      features except the categorical ones. Default: 0.0.
    lambda_loss: Lambda regularization applied to certain training loss
      functions. Only for NDCG loss. Default: 1.0.
    loss: The loss optimized by the model. If not specified (DEFAULT) the loss
      is selected automatically according to the \\"task\\" and label
      statistics. For example, if task=CLASSIFICATION and the label has two
      possible values, the loss will be set to BINOMIAL_LOG_LIKELIHOOD. Possible
      values are: - `DEFAULT`: Select the loss automatically according to the
      task and label statistics. - `BINOMIAL_LOG_LIKELIHOOD`: Binomial log
      likelihood. Only valid for binary classification. - `SQUARED_ERROR`: Least
      square loss. Only valid for regression. - `POISSON`: Poisson log
      likelihood loss. Mainly used for counting problems. Only valid for
      regression. - `MULTINOMIAL_LOG_LIKELIHOOD`: Multinomial log likelihood
      i.e. cross-entropy. Only valid for binary or multi-class classification. -
      `LAMBDA_MART_NDCG5`: LambdaMART with NDCG5. - `XE_NDCG_MART`:  Cross
      Entropy Loss NDCG. See arxiv.org/abs/1911.09798. - `BINARY_FOCAL_LOSS`:
      Focal loss. Only valid for binary classification. See
      https://arxiv.org/pdf/1708.02002.pdf. - `POISSON`: Poisson log likelihood.
        Only valid for regression. - `MEAN_AVERAGE_ERROR`: Mean average error
        a.k.a. MAE.
        Default: "DEFAULT".
    max_depth: Maximum depth of the tree. `max_depth=1` means that all trees
      will be roots. `max_depth=-1` means that tree depth is not restricted by
      this parameter. Values <= -2 will be ignored. Default: 6.
    max_num_nodes: Maximum number of nodes in the tree. Set to -1 to disable
      this limit. Only available for `growing_strategy=BEST_FIRST_GLOBAL`.
      Default: None.
    maximum_model_size_in_memory_in_bytes: Limit the size of the model when
      stored in ram. Different algorithms can enforce this limit differently.
      Note that when models are compiled into an inference, the size of the
      inference engine is generally much smaller than the original model.
      Default: -1.0.
    maximum_training_duration_seconds: Maximum training duration of the model
      expressed in seconds. Each learning algorithm is free to use this
      parameter at it sees fit. Enabling maximum training duration makes the
      model training non-deterministic. Default: -1.0.
    min_examples: Minimum number of examples in a node. Default: 5.
    missing_value_policy: Method used to handle missing attribute values. -
      `GLOBAL_IMPUTATION`: Missing attribute values are imputed, with the mean
      (in case of numerical attribute) or the most-frequent-item (in case of
      categorical attribute) computed on the entire dataset (i.e. the
      information contained in the data spec). - `LOCAL_IMPUTATION`: Missing
      attribute values are imputed with the mean (numerical attribute) or
      most-frequent-item (in the case of categorical attribute) evaluated on the
      training examples in the current node. - `RANDOM_LOCAL_IMPUTATION`:
      Missing attribute values are imputed from randomly sampled values from the
      training examples in the current node. This method was proposed by Clinic
      et al. in "Random Survival Forests"
      (https://projecteuclid.org/download/pdfview_1/euclid.aoas/1223908043).
        Default: "GLOBAL_IMPUTATION".
    num_candidate_attributes: Number of unique valid attributes tested for each
      node. An attribute is valid if it has at least a valid split. If
      `num_candidate_attributes=0`, the value is set to the classical default
      value for Random Forest: `sqrt(number of input attributes)` in case of
      classification and `number_of_input_attributes / 3` in case of regression.
      If `num_candidate_attributes=-1`, all the attributes are tested. Default:
      -1.
    num_candidate_attributes_ratio: Ratio of attributes tested at each node. If
      set, it is equivalent to `num_candidate_attributes =
      number_of_input_features x num_candidate_attributes_ratio`. The possible
      values are between ]0, and 1] as well as -1. If not set or equal to -1,
      the `num_candidate_attributes` is used. Default: -1.0.
    num_trees: Maximum number of decision trees. The effective number of trained
      tree can be smaller if early stopping is enabled. Default: 300.
    pure_serving_model: Clear the model from any information that is not
      required for model serving. This includes debugging, model interpretation
      and other meta-data. The size of the serialized model can be reduced
      significatively (50% model size reduction is common). This parameter has
      no impact on the quality, serving speed or RAM usage of model serving.
      Default: False.
    random_seed: Random seed for the training of the model. Learners are
      expected to be deterministic by the random seed. Default: 123456.
    sampling_method: Control the sampling of the datasets used to train
      individual trees. - NONE: No sampling is applied. This is equivalent to
      RANDOM sampling with \\"subsample=1\\". - RANDOM (default): Uniform random
      sampling. Automatically selected if "subsample" is set. - GOSS:
      Gradient-based One-Side Sampling. Automatically selected if "goss_alpha"
      or "goss_beta" is set. - SELGB: Selective Gradient Boosting. Automatically
      selected if "selective_gradient_boosting_ratio" is set. Only valid for
      ranking.
        Default: "RANDOM".
    selective_gradient_boosting_ratio: Ratio of the dataset used to train
      individual tree for the selective Gradient Boosting (Selective Gradient
      Boosting for Effective Learning to Rank; Lucchese et al;
      http://quickrank.isti.cnr.it/selective-data/selective-SIGIR2018.pdf)
        sampling method. Default: 0.01.
    shrinkage: Coefficient applied to each tree prediction. A small value (0.02)
      tends to give more accurate results (assuming enough trees are trained),
      but results in larger models. Analogous to neural network learning rate.
      Default: 0.1.
    sorting_strategy: How are sorted the numerical features in order to find the
      splits - PRESORT: The features are pre-sorted at the start of the
      training. This solution is faster but consumes much more memory than
      IN_NODE. - IN_NODE: The features are sorted just before being used in the
      node. This solution is slow but consumes little amount of memory. .
      Default: "PRESORT".
    sparse_oblique_max_num_projections: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Maximum number of projections (applied after
      the num_projections_exponent). Oblique splits try out
      max(p^num_projections_exponent, max_num_projections) random projections
      for choosing a split, where p is the number of numerical features.
      Increasing "max_num_projections" increases the training time but not the
      inference time. In late stage model development, if every bit of accuracy
      if important, increase this value. The paper "Sparse Projection Oblique
      Random Forests" (Tomita et al, 2020) does not define this hyperparameter.
      Default: None.
    sparse_oblique_normalization: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Normalization applied on the features, before
      applying the sparse oblique projections. - `NONE`: No normalization. -
      `STANDARD_DEVIATION`: Normalize the feature by the estimated standard
      deviation on the entire train dataset. Also known as Z-Score
      normalization. - `MIN_MAX`: Normalize the feature by the range (i.e.
      max-min) estimated on the entire train dataset. Default: None.
    sparse_oblique_num_projections_exponent: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Controls of the number of random projections
      to test at each node. Increasing this value very likely improves the
      quality of the model, drastically increases the training time, and doe not
      impact the inference time. Oblique splits try out
      max(p^num_projections_exponent, max_num_projections) random projections
      for choosing a split, where p is the number of numerical features.
      Therefore, increasing this `num_projections_exponent` and possibly
      `max_num_projections` may improve model quality, but will also
      significantly increase training time. Note that the complexity of
      (classic) Random Forests is roughly proportional to
      `num_projections_exponent=0.5`, since it considers sqrt(num_features) for
      a split. The complexity of (classic) GBDT is roughly proportional to
      `num_projections_exponent=1`, since it considers all features for a split.
      The paper "Sparse Projection Oblique Random Forests" (Tomita et al, 2020)
      recommends values in [1/4, 2]. Default: None.
    sparse_oblique_projection_density_factor: Density of the projections as an
      exponent of the number of features. Independently for each projection,
      each feature has a probability "projection_density_factor / num_features"
      to be considered in the projection. The paper "Sparse Projection Oblique
      Random Forests" (Tomita et al, 2020) calls this parameter `lambda` and
      recommends values in [1, 5]. Increasing this value increases training and
      inference time (on average). This value is best tuned for each dataset.
      Default: None.
    sparse_oblique_weights: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Possible values: - `BINARY`: The oblique
      weights are sampled in {-1,1} (default). - `CONTINUOUS`: The oblique
      weights are be sampled in [-1,1]. Default: None.
    split_axis: What structure of split to consider for numerical features. -
      `AXIS_ALIGNED`: Axis aligned splits (i.e. one condition at a time). This
      is the "classical" way to train a tree. Default value. - `SPARSE_OBLIQUE`:
      Sparse oblique splits (i.e. splits one a small number of features) from
      "Sparse Projection Oblique Random Forests", Tomita et al., 2020. Default:
      "AXIS_ALIGNED".
    subsample: Ratio of the dataset (sampling without replacement) used to train
      individual trees for the random sampling method. If \\"subsample\\" is set
      and if \\"sampling_method\\" is NOT set or set to \\"NONE\\", then
      \\"sampling_method\\" is implicitly set to \\"RANDOM\\". In other words,
      to enable random subsampling, you only need to set "\\"subsample\\".
      Default: 1.0.
    uplift_min_examples_in_treatment: For uplift models only. Minimum number of
      examples per treatment in a node. Default: 5.
    uplift_split_score: For uplift models only. Splitter score i.e. score
      optimized by the splitters. The scores are introduced in "Decision trees
      for uplift modeling with single and multiple treatments", Rzepakowski et
      al. Notation: `p` probability / average value of the positive outcome, `q`
      probability / average value in the control group. - `KULLBACK_LEIBLER` or
      `KL`: - p log (p/q) - `EUCLIDEAN_DISTANCE` or `ED`: (p-q)^2 -
      `CHI_SQUARED` or `CS`: (p-q)^2/q
        Default: "KULLBACK_LEIBLER".
    use_hessian_gain: Use true, uses a formulation of split gain with a hessian
      term i.e. optimizes the splits to minimize the variance of "gradient /
      hessian. Available for all losses except regression. Default: False.
    validation_interval_in_trees: Evaluate the model on the validation set every
      "validation_interval_in_trees" trees. Increasing this value reduce the
      cost of validation and can impact the early stopping policy (as early
      stopping is only tested during the validation). Default: 1.
    validation_ratio: Fraction of the training dataset used for validation if
      not validation dataset is provided. The validation dataset, whether
      provided directly or extracted from the training dataset, is used to
      compute the validation loss, other validation metrics, and possibly
      trigger early stopping (if enabled). When early stopping is disabled, the
      validation dataset is only used for monitoring and does not influence the
      model directly. If the "validation_ratio" is set to 0, early stopping is
      disabled (i.e., it implies setting early_stopping=NONE). Default: 0.1.
  """

  @core._list_explicit_arguments
  def __init__(
      self,
      task: Optional[TaskType] = core.Task.CLASSIFICATION,
      features: Optional[List[core.FeatureUsage]] = None,
      exclude_non_specified_features: Optional[bool] = False,
      preprocessing: Optional["tf_keras.models.Functional"] = None,
      postprocessing: Optional["tf_keras.models.Functional"] = None,
      training_preprocessing: Optional["tf_keras.models.Functional"] = None,
      ranking_group: Optional[str] = None,
      uplift_treatment: Optional[str] = None,
      temp_directory: Optional[str] = None,
      verbose: int = 1,
      hyperparameter_template: Optional[str] = None,
      advanced_arguments: Optional[AdvancedArguments] = None,
      num_threads: Optional[int] = None,
      name: Optional[str] = None,
      max_vocab_count: Optional[int] = 2000,
      try_resume_training: Optional[bool] = True,
      check_dataset: Optional[bool] = True,
      tuner: Optional[tuner_lib.Tuner] = None,
      discretize_numerical_features: bool = False,
      num_discretized_numerical_bins: int = 255,
      multitask: Optional[List[MultiTaskItem]] = None,
      adapt_subsample_for_maximum_training_duration: Optional[bool] = False,
      allow_na_conditions: Optional[bool] = False,
      apply_link_function: Optional[bool] = True,
      categorical_algorithm: Optional[str] = "CART",
      categorical_set_split_greedy_sampling: Optional[float] = 0.1,
      categorical_set_split_max_num_items: Optional[int] = -1,
      categorical_set_split_min_item_frequency: Optional[int] = 1,
      compute_permutation_variable_importance: Optional[bool] = False,
      dart_dropout: Optional[float] = 0.01,
      early_stopping: Optional[str] = "LOSS_INCREASE",
      early_stopping_initial_iteration: Optional[int] = 10,
      early_stopping_num_trees_look_ahead: Optional[int] = 30,
      focal_loss_alpha: Optional[float] = 0.5,
      focal_loss_gamma: Optional[float] = 2.0,
      forest_extraction: Optional[str] = "MART",
      goss_alpha: Optional[float] = 0.2,
      goss_beta: Optional[float] = 0.1,
      growing_strategy: Optional[str] = "LOCAL",
      honest: Optional[bool] = False,
      honest_fixed_separation: Optional[bool] = False,
      honest_ratio_leaf_examples: Optional[float] = 0.5,
      in_split_min_examples_check: Optional[bool] = True,
      keep_non_leaf_label_distribution: Optional[bool] = True,
      l1_regularization: Optional[float] = 0.0,
      l2_categorical_regularization: Optional[float] = 1.0,
      l2_regularization: Optional[float] = 0.0,
      lambda_loss: Optional[float] = 1.0,
      loss: Optional[str] = "DEFAULT",
      max_depth: Optional[int] = 6,
      max_num_nodes: Optional[int] = None,
      maximum_model_size_in_memory_in_bytes: Optional[float] = -1.0,
      maximum_training_duration_seconds: Optional[float] = -1.0,
      min_examples: Optional[int] = 5,
      missing_value_policy: Optional[str] = "GLOBAL_IMPUTATION",
      num_candidate_attributes: Optional[int] = -1,
      num_candidate_attributes_ratio: Optional[float] = -1.0,
      num_trees: Optional[int] = 300,
      pure_serving_model: Optional[bool] = False,
      random_seed: Optional[int] = 123456,
      sampling_method: Optional[str] = "RANDOM",
      selective_gradient_boosting_ratio: Optional[float] = 0.01,
      shrinkage: Optional[float] = 0.1,
      sorting_strategy: Optional[str] = "PRESORT",
      sparse_oblique_max_num_projections: Optional[int] = None,
      sparse_oblique_normalization: Optional[str] = None,
      sparse_oblique_num_projections_exponent: Optional[float] = None,
      sparse_oblique_projection_density_factor: Optional[float] = None,
      sparse_oblique_weights: Optional[str] = None,
      split_axis: Optional[str] = "AXIS_ALIGNED",
      subsample: Optional[float] = 1.0,
      uplift_min_examples_in_treatment: Optional[int] = 5,
      uplift_split_score: Optional[str] = "KULLBACK_LEIBLER",
      use_hessian_gain: Optional[bool] = False,
      validation_interval_in_trees: Optional[int] = 1,
      validation_ratio: Optional[float] = 0.1,
      explicit_args: Optional[Set[str]] = None,
  ):

    learner_params = {
        "adapt_subsample_for_maximum_training_duration": (
            adapt_subsample_for_maximum_training_duration
        ),
        "allow_na_conditions": allow_na_conditions,
        "apply_link_function": apply_link_function,
        "categorical_algorithm": categorical_algorithm,
        "categorical_set_split_greedy_sampling": (
            categorical_set_split_greedy_sampling
        ),
        "categorical_set_split_max_num_items": (
            categorical_set_split_max_num_items
        ),
        "categorical_set_split_min_item_frequency": (
            categorical_set_split_min_item_frequency
        ),
        "compute_permutation_variable_importance": (
            compute_permutation_variable_importance
        ),
        "dart_dropout": dart_dropout,
        "early_stopping": early_stopping,
        "early_stopping_initial_iteration": early_stopping_initial_iteration,
        "early_stopping_num_trees_look_ahead": (
            early_stopping_num_trees_look_ahead
        ),
        "focal_loss_alpha": focal_loss_alpha,
        "focal_loss_gamma": focal_loss_gamma,
        "forest_extraction": forest_extraction,
        "goss_alpha": goss_alpha,
        "goss_beta": goss_beta,
        "growing_strategy": growing_strategy,
        "honest": honest,
        "honest_fixed_separation": honest_fixed_separation,
        "honest_ratio_leaf_examples": honest_ratio_leaf_examples,
        "in_split_min_examples_check": in_split_min_examples_check,
        "keep_non_leaf_label_distribution": keep_non_leaf_label_distribution,
        "l1_regularization": l1_regularization,
        "l2_categorical_regularization": l2_categorical_regularization,
        "l2_regularization": l2_regularization,
        "lambda_loss": lambda_loss,
        "loss": loss,
        "max_depth": max_depth,
        "max_num_nodes": max_num_nodes,
        "maximum_model_size_in_memory_in_bytes": (
            maximum_model_size_in_memory_in_bytes
        ),
        "maximum_training_duration_seconds": maximum_training_duration_seconds,
        "min_examples": min_examples,
        "missing_value_policy": missing_value_policy,
        "num_candidate_attributes": num_candidate_attributes,
        "num_candidate_attributes_ratio": num_candidate_attributes_ratio,
        "num_trees": num_trees,
        "pure_serving_model": pure_serving_model,
        "random_seed": random_seed,
        "sampling_method": sampling_method,
        "selective_gradient_boosting_ratio": selective_gradient_boosting_ratio,
        "shrinkage": shrinkage,
        "sorting_strategy": sorting_strategy,
        "sparse_oblique_max_num_projections": (
            sparse_oblique_max_num_projections
        ),
        "sparse_oblique_normalization": sparse_oblique_normalization,
        "sparse_oblique_num_projections_exponent": (
            sparse_oblique_num_projections_exponent
        ),
        "sparse_oblique_projection_density_factor": (
            sparse_oblique_projection_density_factor
        ),
        "sparse_oblique_weights": sparse_oblique_weights,
        "split_axis": split_axis,
        "subsample": subsample,
        "uplift_min_examples_in_treatment": uplift_min_examples_in_treatment,
        "uplift_split_score": uplift_split_score,
        "use_hessian_gain": use_hessian_gain,
        "validation_interval_in_trees": validation_interval_in_trees,
        "validation_ratio": validation_ratio,
    }

    if hyperparameter_template is not None:
      learner_params = core._apply_hp_template(
          learner_params,
          hyperparameter_template,
          self.predefined_hyperparameters(),
          explicit_args,
      )

    super(GradientBoostedTreesModel, self).__init__(
        task=task,
        learner="GRADIENT_BOOSTED_TREES",
        learner_params=learner_params,
        features=features,
        exclude_non_specified_features=exclude_non_specified_features,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        training_preprocessing=training_preprocessing,
        ranking_group=ranking_group,
        uplift_treatment=uplift_treatment,
        temp_directory=temp_directory,
        verbose=verbose,
        advanced_arguments=advanced_arguments,
        num_threads=num_threads,
        name=name,
        max_vocab_count=max_vocab_count,
        try_resume_training=try_resume_training,
        check_dataset=check_dataset,
        tuner=tuner,
        discretize_numerical_features=discretize_numerical_features,
        num_discretized_numerical_bins=num_discretized_numerical_bins,
        multitask=multitask,
    )

  @staticmethod
  def predefined_hyperparameters() -> List[core.HyperParameterTemplate]:
    return [
        core.HyperParameterTemplate(
            name="better_default",
            version=1,
            description=(
                "A configuration that is generally better than the default"
                " parameters without being more expensive."
            ),
            parameters={"growing_strategy": "BEST_FIRST_GLOBAL"},
        ),
        core.HyperParameterTemplate(
            name="benchmark_rank1",
            version=1,
            description=(
                "Top ranking hyper-parameters on our benchmark slightly"
                " modified to run in reasonable time."
            ),
            parameters={
                "growing_strategy": "BEST_FIRST_GLOBAL",
                "categorical_algorithm": "RANDOM",
                "split_axis": "SPARSE_OBLIQUE",
                "sparse_oblique_normalization": "MIN_MAX",
                "sparse_oblique_num_projections_exponent": 1.0,
            },
        ),
    ]

  @staticmethod
  def capabilities() -> abstract_learner_pb2.LearnerCapabilities:
    return abstract_learner_pb2.LearnerCapabilities(
        support_partial_cache_dataset_format=False
    )

class HyperparameterOptimizerModel(core.CoreModel):
  r"""Hyperparameter Optimizer learning algorithm.

  Usage example:

  ```python
  import tensorflow_decision_forests as tfdf
  import pandas as pd

  dataset = pd.read_csv("project/dataset.csv")
  tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="my_label")

  model = tfdf.keras.HyperparameterOptimizerModel()
  model.fit(tf_dataset)

  print(model.summary())
  ```

  Hyper-parameter tuning:

  ```python
  import tensorflow_decision_forests as tfdf
  import pandas as pd

  dataset = pd.read_csv("project/dataset.csv")
  tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="my_label")

  tuner = tfdf.tuner.RandomSearch(num_trials=20)

  # Hyper-parameters to optimize.
  tuner.discret("max_depth", [4, 5, 6, 7])

  model = tfdf.keras.HyperparameterOptimizerModel(tuner=tuner)
  model.fit(tf_dataset)

  print(model.summary())
  ```


  Attributes:
    task: Task to solve (e.g. Task.CLASSIFICATION, Task.REGRESSION,
      Task.RANKING, Task.CATEGORICAL_UPLIFT, Task.NUMERICAL_UPLIFT).
    features: Specify the list and semantic of the input features of the model.
      If not specified, all the available features will be used. If specified
      and if `exclude_non_specified_features=True`, only the features in
      `features` will be used by the model. If "preprocessing" is used,
      `features` corresponds to the output of the preprocessing. In this case,
      it is recommended for the preprocessing to return a dictionary of tensors.
    exclude_non_specified_features: If true, only use the features specified in
      `features`.
    preprocessing: Functional keras model or @tf.function to apply on the input
      feature before the model to train. This preprocessing model can consume
      and return tensors, list of tensors or dictionary of tensors. If
      specified, the model only "sees" the output of the preprocessing (and not
      the raw input). Can be used to prepare the features or to stack multiple
      models on top of each other. Unlike preprocessing done in the tf.dataset,
      the operation in "preprocessing" are serialized with the model.
    postprocessing: Like "preprocessing" but applied on the model output.
    training_preprocessing: Functional keras model or `@tf.function` to apply on
      the input feature, labels, and sample_weight before model training.
    ranking_group: Only for `task=Task.RANKING`. Name of a tf.string feature
      that identifies queries in a query/document ranking task. The ranking
      group is not added automatically for the set of features if
      `exclude_non_specified_features=false`.
    uplift_treatment: Only for task=Task.CATEGORICAL_UPLIFT or
      task=Task.NUMERICAL_UPLIFT. Name of an integer feature that identifies the
      treatment in an uplift problem. The value 0 is reserved for the control
      treatment.
    temp_directory: Temporary directory used to store the model Assets after the
      training, and possibly as a work directory during the training. This
      temporary directory is necessary for the model to be exported after
      training e.g. `model.save(path)`. If not specified, `temp_directory` is
      set to a temporary directory using `tempfile.TemporaryDirectory`. This
      directory is deleted when the model python object is garbage-collected.
    verbose: Verbosity mode. 0 = silent, 1 = small details, 2 = full details.
    hyperparameter_template: Override the default value of the hyper-parameters.
      If None (default) the default parameters of the library are used. If set,
      `default_hyperparameter_template` refers to one of the following
      preconfigured hyper-parameter sets. Those sets outperforms the default
      hyper-parameters (either generally or in specific scenarios). You can omit
      the version (e.g. remove "@v5") to use the last version of the template.
      In this case, the hyper-parameter can change in between releases (not
      recommended for training in production).
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
    multitask: If set, train a multi-task model, that is a model with multiple
      outputs trained to predict different labels. If set, the tf.dataset label
      (i.e. the second selement of the dataset) should be a dictionary of
      label_key:label_values. Only one of `multitask` and `task` can be set.
    maximum_model_size_in_memory_in_bytes: Limit the size of the model when
      stored in ram. Different algorithms can enforce this limit differently.
      Note that when models are compiled into an inference, the size of the
      inference engine is generally much smaller than the original model.
      Default: -1.0.
    maximum_training_duration_seconds: Maximum training duration of the model
      expressed in seconds. Each learning algorithm is free to use this
      parameter at it sees fit. Enabling maximum training duration makes the
      model training non-deterministic. Default: -1.0.
    pure_serving_model: Clear the model from any information that is not
      required for model serving. This includes debugging, model interpretation
      and other meta-data. The size of the serialized model can be reduced
      significatively (50% model size reduction is common). This parameter has
      no impact on the quality, serving speed or RAM usage of model serving.
      Default: False.
    random_seed: Random seed for the training of the model. Learners are
      expected to be deterministic by the random seed. Default: 123456.
  """

  @core._list_explicit_arguments
  def __init__(
      self,
      task: Optional[TaskType] = core.Task.CLASSIFICATION,
      features: Optional[List[core.FeatureUsage]] = None,
      exclude_non_specified_features: Optional[bool] = False,
      preprocessing: Optional["tf_keras.models.Functional"] = None,
      postprocessing: Optional["tf_keras.models.Functional"] = None,
      training_preprocessing: Optional["tf_keras.models.Functional"] = None,
      ranking_group: Optional[str] = None,
      uplift_treatment: Optional[str] = None,
      temp_directory: Optional[str] = None,
      verbose: int = 1,
      hyperparameter_template: Optional[str] = None,
      advanced_arguments: Optional[AdvancedArguments] = None,
      num_threads: Optional[int] = None,
      name: Optional[str] = None,
      max_vocab_count: Optional[int] = 2000,
      try_resume_training: Optional[bool] = True,
      check_dataset: Optional[bool] = True,
      tuner: Optional[tuner_lib.Tuner] = None,
      discretize_numerical_features: bool = False,
      num_discretized_numerical_bins: int = 255,
      multitask: Optional[List[MultiTaskItem]] = None,
      maximum_model_size_in_memory_in_bytes: Optional[float] = -1.0,
      maximum_training_duration_seconds: Optional[float] = -1.0,
      pure_serving_model: Optional[bool] = False,
      random_seed: Optional[int] = 123456,
      explicit_args: Optional[Set[str]] = None,
  ):

    learner_params = {
        "maximum_model_size_in_memory_in_bytes": (
            maximum_model_size_in_memory_in_bytes
        ),
        "maximum_training_duration_seconds": maximum_training_duration_seconds,
        "pure_serving_model": pure_serving_model,
        "random_seed": random_seed,
    }

    if hyperparameter_template is not None:
      learner_params = core._apply_hp_template(
          learner_params,
          hyperparameter_template,
          self.predefined_hyperparameters(),
          explicit_args,
      )

    super(HyperparameterOptimizerModel, self).__init__(
        task=task,
        learner="HYPERPARAMETER_OPTIMIZER",
        learner_params=learner_params,
        features=features,
        exclude_non_specified_features=exclude_non_specified_features,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        training_preprocessing=training_preprocessing,
        ranking_group=ranking_group,
        uplift_treatment=uplift_treatment,
        temp_directory=temp_directory,
        verbose=verbose,
        advanced_arguments=advanced_arguments,
        num_threads=num_threads,
        name=name,
        max_vocab_count=max_vocab_count,
        try_resume_training=try_resume_training,
        check_dataset=check_dataset,
        tuner=tuner,
        discretize_numerical_features=discretize_numerical_features,
        num_discretized_numerical_bins=num_discretized_numerical_bins,
        multitask=multitask,
    )

  @staticmethod
  def predefined_hyperparameters() -> List[core.HyperParameterTemplate]:
    return []

  @staticmethod
  def capabilities() -> abstract_learner_pb2.LearnerCapabilities:
    return abstract_learner_pb2.LearnerCapabilities(
        support_partial_cache_dataset_format=False
    )

class MultitaskerModel(core.CoreModel):
  r"""Multitasker learning algorithm.

  Usage example:

  ```python
  import tensorflow_decision_forests as tfdf
  import pandas as pd

  dataset = pd.read_csv("project/dataset.csv")
  tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="my_label")

  model = tfdf.keras.MultitaskerModel()
  model.fit(tf_dataset)

  print(model.summary())
  ```

  Hyper-parameter tuning:

  ```python
  import tensorflow_decision_forests as tfdf
  import pandas as pd

  dataset = pd.read_csv("project/dataset.csv")
  tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="my_label")

  tuner = tfdf.tuner.RandomSearch(num_trials=20)

  # Hyper-parameters to optimize.
  tuner.discret("max_depth", [4, 5, 6, 7])

  model = tfdf.keras.MultitaskerModel(tuner=tuner)
  model.fit(tf_dataset)

  print(model.summary())
  ```


  Attributes:
    task: Task to solve (e.g. Task.CLASSIFICATION, Task.REGRESSION,
      Task.RANKING, Task.CATEGORICAL_UPLIFT, Task.NUMERICAL_UPLIFT).
    features: Specify the list and semantic of the input features of the model.
      If not specified, all the available features will be used. If specified
      and if `exclude_non_specified_features=True`, only the features in
      `features` will be used by the model. If "preprocessing" is used,
      `features` corresponds to the output of the preprocessing. In this case,
      it is recommended for the preprocessing to return a dictionary of tensors.
    exclude_non_specified_features: If true, only use the features specified in
      `features`.
    preprocessing: Functional keras model or @tf.function to apply on the input
      feature before the model to train. This preprocessing model can consume
      and return tensors, list of tensors or dictionary of tensors. If
      specified, the model only "sees" the output of the preprocessing (and not
      the raw input). Can be used to prepare the features or to stack multiple
      models on top of each other. Unlike preprocessing done in the tf.dataset,
      the operation in "preprocessing" are serialized with the model.
    postprocessing: Like "preprocessing" but applied on the model output.
    training_preprocessing: Functional keras model or `@tf.function` to apply on
      the input feature, labels, and sample_weight before model training.
    ranking_group: Only for `task=Task.RANKING`. Name of a tf.string feature
      that identifies queries in a query/document ranking task. The ranking
      group is not added automatically for the set of features if
      `exclude_non_specified_features=false`.
    uplift_treatment: Only for task=Task.CATEGORICAL_UPLIFT or
      task=Task.NUMERICAL_UPLIFT. Name of an integer feature that identifies the
      treatment in an uplift problem. The value 0 is reserved for the control
      treatment.
    temp_directory: Temporary directory used to store the model Assets after the
      training, and possibly as a work directory during the training. This
      temporary directory is necessary for the model to be exported after
      training e.g. `model.save(path)`. If not specified, `temp_directory` is
      set to a temporary directory using `tempfile.TemporaryDirectory`. This
      directory is deleted when the model python object is garbage-collected.
    verbose: Verbosity mode. 0 = silent, 1 = small details, 2 = full details.
    hyperparameter_template: Override the default value of the hyper-parameters.
      If None (default) the default parameters of the library are used. If set,
      `default_hyperparameter_template` refers to one of the following
      preconfigured hyper-parameter sets. Those sets outperforms the default
      hyper-parameters (either generally or in specific scenarios). You can omit
      the version (e.g. remove "@v5") to use the last version of the template.
      In this case, the hyper-parameter can change in between releases (not
      recommended for training in production).
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
    multitask: If set, train a multi-task model, that is a model with multiple
      outputs trained to predict different labels. If set, the tf.dataset label
      (i.e. the second selement of the dataset) should be a dictionary of
      label_key:label_values. Only one of `multitask` and `task` can be set.
    maximum_model_size_in_memory_in_bytes: Limit the size of the model when
      stored in ram. Different algorithms can enforce this limit differently.
      Note that when models are compiled into an inference, the size of the
      inference engine is generally much smaller than the original model.
      Default: -1.0.
    maximum_training_duration_seconds: Maximum training duration of the model
      expressed in seconds. Each learning algorithm is free to use this
      parameter at it sees fit. Enabling maximum training duration makes the
      model training non-deterministic. Default: -1.0.
    pure_serving_model: Clear the model from any information that is not
      required for model serving. This includes debugging, model interpretation
      and other meta-data. The size of the serialized model can be reduced
      significatively (50% model size reduction is common). This parameter has
      no impact on the quality, serving speed or RAM usage of model serving.
      Default: False.
    random_seed: Random seed for the training of the model. Learners are
      expected to be deterministic by the random seed. Default: 123456.
  """

  @core._list_explicit_arguments
  def __init__(
      self,
      task: Optional[TaskType] = core.Task.CLASSIFICATION,
      features: Optional[List[core.FeatureUsage]] = None,
      exclude_non_specified_features: Optional[bool] = False,
      preprocessing: Optional["tf_keras.models.Functional"] = None,
      postprocessing: Optional["tf_keras.models.Functional"] = None,
      training_preprocessing: Optional["tf_keras.models.Functional"] = None,
      ranking_group: Optional[str] = None,
      uplift_treatment: Optional[str] = None,
      temp_directory: Optional[str] = None,
      verbose: int = 1,
      hyperparameter_template: Optional[str] = None,
      advanced_arguments: Optional[AdvancedArguments] = None,
      num_threads: Optional[int] = None,
      name: Optional[str] = None,
      max_vocab_count: Optional[int] = 2000,
      try_resume_training: Optional[bool] = True,
      check_dataset: Optional[bool] = True,
      tuner: Optional[tuner_lib.Tuner] = None,
      discretize_numerical_features: bool = False,
      num_discretized_numerical_bins: int = 255,
      multitask: Optional[List[MultiTaskItem]] = None,
      maximum_model_size_in_memory_in_bytes: Optional[float] = -1.0,
      maximum_training_duration_seconds: Optional[float] = -1.0,
      pure_serving_model: Optional[bool] = False,
      random_seed: Optional[int] = 123456,
      explicit_args: Optional[Set[str]] = None,
  ):

    learner_params = {
        "maximum_model_size_in_memory_in_bytes": (
            maximum_model_size_in_memory_in_bytes
        ),
        "maximum_training_duration_seconds": maximum_training_duration_seconds,
        "pure_serving_model": pure_serving_model,
        "random_seed": random_seed,
    }

    if hyperparameter_template is not None:
      learner_params = core._apply_hp_template(
          learner_params,
          hyperparameter_template,
          self.predefined_hyperparameters(),
          explicit_args,
      )

    super(MultitaskerModel, self).__init__(
        task=task,
        learner="MULTITASKER",
        learner_params=learner_params,
        features=features,
        exclude_non_specified_features=exclude_non_specified_features,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        training_preprocessing=training_preprocessing,
        ranking_group=ranking_group,
        uplift_treatment=uplift_treatment,
        temp_directory=temp_directory,
        verbose=verbose,
        advanced_arguments=advanced_arguments,
        num_threads=num_threads,
        name=name,
        max_vocab_count=max_vocab_count,
        try_resume_training=try_resume_training,
        check_dataset=check_dataset,
        tuner=tuner,
        discretize_numerical_features=discretize_numerical_features,
        num_discretized_numerical_bins=num_discretized_numerical_bins,
        multitask=multitask,
    )

  @staticmethod
  def predefined_hyperparameters() -> List[core.HyperParameterTemplate]:
    return []

  @staticmethod
  def capabilities() -> abstract_learner_pb2.LearnerCapabilities:
    return abstract_learner_pb2.LearnerCapabilities(
        support_partial_cache_dataset_format=False
    )

class RandomForestModel(core.CoreModel):
  r"""Random Forest learning algorithm.

  A Random Forest (https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
  is a collection of deep CART decision trees trained independently and without
  pruning. Each tree is trained on a random subset of the original training
  dataset (sampled with replacement).

  The algorithm is unique in that it is robust to overfitting, even in extreme
  cases e.g. when there are more features than training examples.

  It is probably the most well-known of the Decision Forest training
  algorithms.

  Usage example:

  ```python
  import tensorflow_decision_forests as tfdf
  import pandas as pd

  dataset = pd.read_csv("project/dataset.csv")
  tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="my_label")

  model = tfdf.keras.RandomForestModel()
  model.fit(tf_dataset)

  print(model.summary())
  ```

  Hyper-parameter tuning:

  ```python
  import tensorflow_decision_forests as tfdf
  import pandas as pd

  dataset = pd.read_csv("project/dataset.csv")
  tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="my_label")

  tuner = tfdf.tuner.RandomSearch(num_trials=20)

  # Hyper-parameters to optimize.
  tuner.discret("max_depth", [4, 5, 6, 7])

  model = tfdf.keras.RandomForestModel(tuner=tuner)
  model.fit(tf_dataset)

  print(model.summary())
  ```


  Attributes:
    task: Task to solve (e.g. Task.CLASSIFICATION, Task.REGRESSION,
      Task.RANKING, Task.CATEGORICAL_UPLIFT, Task.NUMERICAL_UPLIFT).
    features: Specify the list and semantic of the input features of the model.
      If not specified, all the available features will be used. If specified
      and if `exclude_non_specified_features=True`, only the features in
      `features` will be used by the model. If "preprocessing" is used,
      `features` corresponds to the output of the preprocessing. In this case,
      it is recommended for the preprocessing to return a dictionary of tensors.
    exclude_non_specified_features: If true, only use the features specified in
      `features`.
    preprocessing: Functional keras model or @tf.function to apply on the input
      feature before the model to train. This preprocessing model can consume
      and return tensors, list of tensors or dictionary of tensors. If
      specified, the model only "sees" the output of the preprocessing (and not
      the raw input). Can be used to prepare the features or to stack multiple
      models on top of each other. Unlike preprocessing done in the tf.dataset,
      the operation in "preprocessing" are serialized with the model.
    postprocessing: Like "preprocessing" but applied on the model output.
    training_preprocessing: Functional keras model or `@tf.function` to apply on
      the input feature, labels, and sample_weight before model training.
    ranking_group: Only for `task=Task.RANKING`. Name of a tf.string feature
      that identifies queries in a query/document ranking task. The ranking
      group is not added automatically for the set of features if
      `exclude_non_specified_features=false`.
    uplift_treatment: Only for task=Task.CATEGORICAL_UPLIFT or
      task=Task.NUMERICAL_UPLIFT. Name of an integer feature that identifies the
      treatment in an uplift problem. The value 0 is reserved for the control
      treatment.
    temp_directory: Temporary directory used to store the model Assets after the
      training, and possibly as a work directory during the training. This
      temporary directory is necessary for the model to be exported after
      training e.g. `model.save(path)`. If not specified, `temp_directory` is
      set to a temporary directory using `tempfile.TemporaryDirectory`. This
      directory is deleted when the model python object is garbage-collected.
    verbose: Verbosity mode. 0 = silent, 1 = small details, 2 = full details.
    hyperparameter_template: Override the default value of the hyper-parameters.
      If None (default) the default parameters of the library are used. If set,
      `default_hyperparameter_template` refers to one of the following
      preconfigured hyper-parameter sets. Those sets outperforms the default
      hyper-parameters (either generally or in specific scenarios). You can omit
      the version (e.g. remove "@v5") to use the last version of the template.
      In this case, the hyper-parameter can change in between releases (not
      recommended for training in production). - better_default@v1: A
      configuration that is generally better than the default parameters without
      being more expensive. The parameters are: winner_take_all=True. -
      benchmark_rank1@v1: Top ranking hyper-parameters on our benchmark slightly
      modified to run in reasonable time. The parameters are:
      winner_take_all=True, categorical_algorithm="RANDOM",
      split_axis="SPARSE_OBLIQUE", sparse_oblique_normalization="MIN_MAX",
      sparse_oblique_num_projections_exponent=1.0.
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
    multitask: If set, train a multi-task model, that is a model with multiple
      outputs trained to predict different labels. If set, the tf.dataset label
      (i.e. the second selement of the dataset) should be a dictionary of
      label_key:label_values. Only one of `multitask` and `task` can be set.
    adapt_bootstrap_size_ratio_for_maximum_training_duration: Control how the
      maximum training duration (if set) is applied. If false, the training stop
      when the time is used. If true, adapts the size of the sampled dataset
      used to train each tree such that `num_trees` will train within
      `maximum_training_duration`. Has no effect if there is no maximum training
      duration specified. Default: False.
    allow_na_conditions: If true, the tree training evaluates conditions of the
      type `X is NA` i.e. `X is missing`. Default: False.
    bootstrap_size_ratio: Number of examples used to train each trees; expressed
      as a ratio of the training dataset size. Default: 1.0.
    bootstrap_training_dataset: If true (default), each tree is trained on a
      separate dataset sampled with replacement from the original dataset. If
      false, all the trees are trained on the entire same dataset. If
      bootstrap_training_dataset:false, OOB metrics are not available.
        bootstrap_training_dataset=false is used in "Extremely randomized trees"
        (https://link.springer.com/content/pdf/10.1007%2Fs10994-006-6226-1.pdf).
      Default: True.
    categorical_algorithm: How to learn splits on categorical attributes. -
      `CART`: CART algorithm. Find categorical splits of the form "value \\in
      mask". The solution is exact for binary classification, regression and
      ranking. It is approximated for multi-class classification. This is a good
      first algorithm to use. In case of overfitting (very small dataset, large
      dictionary), the "random" algorithm is a good alternative. - `ONE_HOT`:
      One-hot encoding. Find the optimal categorical split of the form
      "attribute == param". This method is similar (but more efficient) than
      converting converting each possible categorical value into a boolean
      feature. This method is available for comparison purpose and generally
      performs worse than other alternatives. - `RANDOM`: Best splits among a
      set of random candidate. Find the a categorical split of the form "value
      \\in mask" using a random search. This solution can be seen as an
      approximation of the CART algorithm. This method is a strong alternative
      to CART. This algorithm is inspired from section "5.1 Categorical
      Variables" of "Random Forest", 2001.
        Default: "CART".
    categorical_set_split_greedy_sampling: For categorical set splits e.g.
      texts. Probability for a categorical value to be a candidate for the
      positive set. The sampling is applied once per node (i.e. not at every
      step of the greedy optimization). Default: 0.1.
    categorical_set_split_max_num_items: For categorical set splits e.g. texts.
      Maximum number of items (prior to the sampling). If more items are
      available, the least frequent items are ignored. Changing this value is
      similar to change the "max_vocab_count" before loading the dataset, with
      the following exception: With `max_vocab_count`, all the remaining items
      are grouped in a special Out-of-vocabulary item. With `max_num_items`,
      this is not the case. Default: -1.
    categorical_set_split_min_item_frequency: For categorical set splits e.g.
      texts. Minimum number of occurrences of an item to be considered.
      Default: 1.
    compute_oob_performances: If true, compute the Out-of-bag evaluation (then
      available in the summary and model inspector). This evaluation is a cheap
      alternative to cross-validation evaluation. Default: True.
    compute_oob_variable_importances: If true, compute the Out-of-bag feature
      importance (then available in the summary and model inspector). Note that
      the OOB feature importance can be expensive to compute. Default: False.
    growing_strategy: How to grow the tree. - `LOCAL`: Each node is split
      independently of the other nodes. In other words, as long as a node
      satisfy the splits "constraints (e.g. maximum depth, minimum number of
      observations), the node will be split. This is the "classical" way to grow
      decision trees. - `BEST_FIRST_GLOBAL`: The node with the best loss
      reduction among all the nodes of the tree is selected for splitting. This
      method is also called "best first" or "leaf-wise growth". See "Best-first
      decision tree learning", Shi and "Additive logistic regression : A
      statistical view of boosting", Friedman for more details. Default:
      "LOCAL".
    honest: In honest trees, different training examples are used to infer the
      structure and the leaf values. This regularization technique trades
      examples for bias estimates. It might increase or reduce the quality of
      the model. See "Generalized Random Forests", Athey et al. In this paper,
      Honest trees are trained with the Random Forest algorithm with a sampling
      without replacement. Default: False.
    honest_fixed_separation: For honest trees only i.e. honest=true. If true, a
      new random separation is generated for each tree. If false, the same
      separation is used for all the trees (e.g., in Gradient Boosted Trees
      containing multiple trees). Default: False.
    honest_ratio_leaf_examples: For honest trees only i.e. honest=true. Ratio of
      examples used to set the leaf values. Default: 0.5.
    in_split_min_examples_check: Whether to check the `min_examples` constraint
      in the split search (i.e. splits leading to one child having less than
      `min_examples` examples are considered invalid) or before the split search
      (i.e. a node can be derived only if it contains more than `min_examples`
      examples). If false, there can be nodes with less than `min_examples`
      training examples. Default: True.
    keep_non_leaf_label_distribution: Whether to keep the node value (i.e. the
      distribution of the labels of the training examples) of non-leaf nodes.
      This information is not used during serving, however it can be used for
      model interpretation as well as hyper parameter tuning. This can take lots
      of space, sometimes accounting for half of the model size. Default: True.
    max_depth: Maximum depth of the tree. `max_depth=1` means that all trees
      will be roots. `max_depth=-1` means that tree depth is not restricted by
      this parameter. Values <= -2 will be ignored. Default: 16.
    max_num_nodes: Maximum number of nodes in the tree. Set to -1 to disable
      this limit. Only available for `growing_strategy=BEST_FIRST_GLOBAL`.
      Default: None.
    maximum_model_size_in_memory_in_bytes: Limit the size of the model when
      stored in ram. Different algorithms can enforce this limit differently.
      Note that when models are compiled into an inference, the size of the
      inference engine is generally much smaller than the original model.
      Default: -1.0.
    maximum_training_duration_seconds: Maximum training duration of the model
      expressed in seconds. Each learning algorithm is free to use this
      parameter at it sees fit. Enabling maximum training duration makes the
      model training non-deterministic. Default: -1.0.
    min_examples: Minimum number of examples in a node. Default: 5.
    missing_value_policy: Method used to handle missing attribute values. -
      `GLOBAL_IMPUTATION`: Missing attribute values are imputed, with the mean
      (in case of numerical attribute) or the most-frequent-item (in case of
      categorical attribute) computed on the entire dataset (i.e. the
      information contained in the data spec). - `LOCAL_IMPUTATION`: Missing
      attribute values are imputed with the mean (numerical attribute) or
      most-frequent-item (in the case of categorical attribute) evaluated on the
      training examples in the current node. - `RANDOM_LOCAL_IMPUTATION`:
      Missing attribute values are imputed from randomly sampled values from the
      training examples in the current node. This method was proposed by Clinic
      et al. in "Random Survival Forests"
      (https://projecteuclid.org/download/pdfview_1/euclid.aoas/1223908043).
        Default: "GLOBAL_IMPUTATION".
    num_candidate_attributes: Number of unique valid attributes tested for each
      node. An attribute is valid if it has at least a valid split. If
      `num_candidate_attributes=0`, the value is set to the classical default
      value for Random Forest: `sqrt(number of input attributes)` in case of
      classification and `number_of_input_attributes / 3` in case of regression.
      If `num_candidate_attributes=-1`, all the attributes are tested. Default:
      0.
    num_candidate_attributes_ratio: Ratio of attributes tested at each node. If
      set, it is equivalent to `num_candidate_attributes =
      number_of_input_features x num_candidate_attributes_ratio`. The possible
      values are between ]0, and 1] as well as -1. If not set or equal to -1,
      the `num_candidate_attributes` is used. Default: -1.0.
    num_oob_variable_importances_permutations: Number of time the dataset is
      re-shuffled to compute the permutation variable importances. Increasing
      this value increase the training time (if
      "compute_oob_variable_importances:true") as well as the stability of the
      oob variable importance metrics. Default: 1.
    num_trees: Number of individual decision trees. Increasing the number of
      trees can increase the quality of the model at the expense of size,
      training speed, and inference latency. Default: 300.
    pure_serving_model: Clear the model from any information that is not
      required for model serving. This includes debugging, model interpretation
      and other meta-data. The size of the serialized model can be reduced
      significatively (50% model size reduction is common). This parameter has
      no impact on the quality, serving speed or RAM usage of model serving.
      Default: False.
    random_seed: Random seed for the training of the model. Learners are
      expected to be deterministic by the random seed. Default: 123456.
    sampling_with_replacement: If true, the training examples are sampled with
      replacement. If false, the training samples are sampled without
      replacement. Only used when "bootstrap_training_dataset=true". If false
      (sampling without replacement) and if "bootstrap_size_ratio=1" (default),
      all the examples are used to train all the trees (you probably do not want
      that). Default: True.
    sorting_strategy: How are sorted the numerical features in order to find the
      splits - PRESORT: The features are pre-sorted at the start of the
      training. This solution is faster but consumes much more memory than
      IN_NODE. - IN_NODE: The features are sorted just before being used in the
      node. This solution is slow but consumes little amount of memory. .
      Default: "PRESORT".
    sparse_oblique_max_num_projections: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Maximum number of projections (applied after
      the num_projections_exponent). Oblique splits try out
      max(p^num_projections_exponent, max_num_projections) random projections
      for choosing a split, where p is the number of numerical features.
      Increasing "max_num_projections" increases the training time but not the
      inference time. In late stage model development, if every bit of accuracy
      if important, increase this value. The paper "Sparse Projection Oblique
      Random Forests" (Tomita et al, 2020) does not define this hyperparameter.
      Default: None.
    sparse_oblique_normalization: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Normalization applied on the features, before
      applying the sparse oblique projections. - `NONE`: No normalization. -
      `STANDARD_DEVIATION`: Normalize the feature by the estimated standard
      deviation on the entire train dataset. Also known as Z-Score
      normalization. - `MIN_MAX`: Normalize the feature by the range (i.e.
      max-min) estimated on the entire train dataset. Default: None.
    sparse_oblique_num_projections_exponent: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Controls of the number of random projections
      to test at each node. Increasing this value very likely improves the
      quality of the model, drastically increases the training time, and doe not
      impact the inference time. Oblique splits try out
      max(p^num_projections_exponent, max_num_projections) random projections
      for choosing a split, where p is the number of numerical features.
      Therefore, increasing this `num_projections_exponent` and possibly
      `max_num_projections` may improve model quality, but will also
      significantly increase training time. Note that the complexity of
      (classic) Random Forests is roughly proportional to
      `num_projections_exponent=0.5`, since it considers sqrt(num_features) for
      a split. The complexity of (classic) GBDT is roughly proportional to
      `num_projections_exponent=1`, since it considers all features for a split.
      The paper "Sparse Projection Oblique Random Forests" (Tomita et al, 2020)
      recommends values in [1/4, 2]. Default: None.
    sparse_oblique_projection_density_factor: Density of the projections as an
      exponent of the number of features. Independently for each projection,
      each feature has a probability "projection_density_factor / num_features"
      to be considered in the projection. The paper "Sparse Projection Oblique
      Random Forests" (Tomita et al, 2020) calls this parameter `lambda` and
      recommends values in [1, 5]. Increasing this value increases training and
      inference time (on average). This value is best tuned for each dataset.
      Default: None.
    sparse_oblique_weights: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Possible values: - `BINARY`: The oblique
      weights are sampled in {-1,1} (default). - `CONTINUOUS`: The oblique
      weights are be sampled in [-1,1]. Default: None.
    split_axis: What structure of split to consider for numerical features. -
      `AXIS_ALIGNED`: Axis aligned splits (i.e. one condition at a time). This
      is the "classical" way to train a tree. Default value. - `SPARSE_OBLIQUE`:
      Sparse oblique splits (i.e. splits one a small number of features) from
      "Sparse Projection Oblique Random Forests", Tomita et al., 2020. Default:
      "AXIS_ALIGNED".
    uplift_min_examples_in_treatment: For uplift models only. Minimum number of
      examples per treatment in a node. Default: 5.
    uplift_split_score: For uplift models only. Splitter score i.e. score
      optimized by the splitters. The scores are introduced in "Decision trees
      for uplift modeling with single and multiple treatments", Rzepakowski et
      al. Notation: `p` probability / average value of the positive outcome, `q`
      probability / average value in the control group. - `KULLBACK_LEIBLER` or
      `KL`: - p log (p/q) - `EUCLIDEAN_DISTANCE` or `ED`: (p-q)^2 -
      `CHI_SQUARED` or `CS`: (p-q)^2/q
        Default: "KULLBACK_LEIBLER".
    winner_take_all: Control how classification trees vote. If true, each tree
      votes for one class. If false, each tree vote for a distribution of
      classes. winner_take_all_inference=false is often preferable. Default:
      True.
  """

  @core._list_explicit_arguments
  def __init__(
      self,
      task: Optional[TaskType] = core.Task.CLASSIFICATION,
      features: Optional[List[core.FeatureUsage]] = None,
      exclude_non_specified_features: Optional[bool] = False,
      preprocessing: Optional["tf_keras.models.Functional"] = None,
      postprocessing: Optional["tf_keras.models.Functional"] = None,
      training_preprocessing: Optional["tf_keras.models.Functional"] = None,
      ranking_group: Optional[str] = None,
      uplift_treatment: Optional[str] = None,
      temp_directory: Optional[str] = None,
      verbose: int = 1,
      hyperparameter_template: Optional[str] = None,
      advanced_arguments: Optional[AdvancedArguments] = None,
      num_threads: Optional[int] = None,
      name: Optional[str] = None,
      max_vocab_count: Optional[int] = 2000,
      try_resume_training: Optional[bool] = True,
      check_dataset: Optional[bool] = True,
      tuner: Optional[tuner_lib.Tuner] = None,
      discretize_numerical_features: bool = False,
      num_discretized_numerical_bins: int = 255,
      multitask: Optional[List[MultiTaskItem]] = None,
      adapt_bootstrap_size_ratio_for_maximum_training_duration: Optional[
          bool
      ] = False,
      allow_na_conditions: Optional[bool] = False,
      bootstrap_size_ratio: Optional[float] = 1.0,
      bootstrap_training_dataset: Optional[bool] = True,
      categorical_algorithm: Optional[str] = "CART",
      categorical_set_split_greedy_sampling: Optional[float] = 0.1,
      categorical_set_split_max_num_items: Optional[int] = -1,
      categorical_set_split_min_item_frequency: Optional[int] = 1,
      compute_oob_performances: Optional[bool] = True,
      compute_oob_variable_importances: Optional[bool] = False,
      growing_strategy: Optional[str] = "LOCAL",
      honest: Optional[bool] = False,
      honest_fixed_separation: Optional[bool] = False,
      honest_ratio_leaf_examples: Optional[float] = 0.5,
      in_split_min_examples_check: Optional[bool] = True,
      keep_non_leaf_label_distribution: Optional[bool] = True,
      max_depth: Optional[int] = 16,
      max_num_nodes: Optional[int] = None,
      maximum_model_size_in_memory_in_bytes: Optional[float] = -1.0,
      maximum_training_duration_seconds: Optional[float] = -1.0,
      min_examples: Optional[int] = 5,
      missing_value_policy: Optional[str] = "GLOBAL_IMPUTATION",
      num_candidate_attributes: Optional[int] = 0,
      num_candidate_attributes_ratio: Optional[float] = -1.0,
      num_oob_variable_importances_permutations: Optional[int] = 1,
      num_trees: Optional[int] = 300,
      pure_serving_model: Optional[bool] = False,
      random_seed: Optional[int] = 123456,
      sampling_with_replacement: Optional[bool] = True,
      sorting_strategy: Optional[str] = "PRESORT",
      sparse_oblique_max_num_projections: Optional[int] = None,
      sparse_oblique_normalization: Optional[str] = None,
      sparse_oblique_num_projections_exponent: Optional[float] = None,
      sparse_oblique_projection_density_factor: Optional[float] = None,
      sparse_oblique_weights: Optional[str] = None,
      split_axis: Optional[str] = "AXIS_ALIGNED",
      uplift_min_examples_in_treatment: Optional[int] = 5,
      uplift_split_score: Optional[str] = "KULLBACK_LEIBLER",
      winner_take_all: Optional[bool] = True,
      explicit_args: Optional[Set[str]] = None,
  ):

    learner_params = {
        "adapt_bootstrap_size_ratio_for_maximum_training_duration": (
            adapt_bootstrap_size_ratio_for_maximum_training_duration
        ),
        "allow_na_conditions": allow_na_conditions,
        "bootstrap_size_ratio": bootstrap_size_ratio,
        "bootstrap_training_dataset": bootstrap_training_dataset,
        "categorical_algorithm": categorical_algorithm,
        "categorical_set_split_greedy_sampling": (
            categorical_set_split_greedy_sampling
        ),
        "categorical_set_split_max_num_items": (
            categorical_set_split_max_num_items
        ),
        "categorical_set_split_min_item_frequency": (
            categorical_set_split_min_item_frequency
        ),
        "compute_oob_performances": compute_oob_performances,
        "compute_oob_variable_importances": compute_oob_variable_importances,
        "growing_strategy": growing_strategy,
        "honest": honest,
        "honest_fixed_separation": honest_fixed_separation,
        "honest_ratio_leaf_examples": honest_ratio_leaf_examples,
        "in_split_min_examples_check": in_split_min_examples_check,
        "keep_non_leaf_label_distribution": keep_non_leaf_label_distribution,
        "max_depth": max_depth,
        "max_num_nodes": max_num_nodes,
        "maximum_model_size_in_memory_in_bytes": (
            maximum_model_size_in_memory_in_bytes
        ),
        "maximum_training_duration_seconds": maximum_training_duration_seconds,
        "min_examples": min_examples,
        "missing_value_policy": missing_value_policy,
        "num_candidate_attributes": num_candidate_attributes,
        "num_candidate_attributes_ratio": num_candidate_attributes_ratio,
        "num_oob_variable_importances_permutations": (
            num_oob_variable_importances_permutations
        ),
        "num_trees": num_trees,
        "pure_serving_model": pure_serving_model,
        "random_seed": random_seed,
        "sampling_with_replacement": sampling_with_replacement,
        "sorting_strategy": sorting_strategy,
        "sparse_oblique_max_num_projections": (
            sparse_oblique_max_num_projections
        ),
        "sparse_oblique_normalization": sparse_oblique_normalization,
        "sparse_oblique_num_projections_exponent": (
            sparse_oblique_num_projections_exponent
        ),
        "sparse_oblique_projection_density_factor": (
            sparse_oblique_projection_density_factor
        ),
        "sparse_oblique_weights": sparse_oblique_weights,
        "split_axis": split_axis,
        "uplift_min_examples_in_treatment": uplift_min_examples_in_treatment,
        "uplift_split_score": uplift_split_score,
        "winner_take_all": winner_take_all,
    }

    if hyperparameter_template is not None:
      learner_params = core._apply_hp_template(
          learner_params,
          hyperparameter_template,
          self.predefined_hyperparameters(),
          explicit_args,
      )

    super(RandomForestModel, self).__init__(
        task=task,
        learner="RANDOM_FOREST",
        learner_params=learner_params,
        features=features,
        exclude_non_specified_features=exclude_non_specified_features,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        training_preprocessing=training_preprocessing,
        ranking_group=ranking_group,
        uplift_treatment=uplift_treatment,
        temp_directory=temp_directory,
        verbose=verbose,
        advanced_arguments=advanced_arguments,
        num_threads=num_threads,
        name=name,
        max_vocab_count=max_vocab_count,
        try_resume_training=try_resume_training,
        check_dataset=check_dataset,
        tuner=tuner,
        discretize_numerical_features=discretize_numerical_features,
        num_discretized_numerical_bins=num_discretized_numerical_bins,
        multitask=multitask,
    )

  @staticmethod
  def predefined_hyperparameters() -> List[core.HyperParameterTemplate]:
    return [
        core.HyperParameterTemplate(
            name="better_default",
            version=1,
            description=(
                "A configuration that is generally better than the default"
                " parameters without being more expensive."
            ),
            parameters={"winner_take_all": True},
        ),
        core.HyperParameterTemplate(
            name="benchmark_rank1",
            version=1,
            description=(
                "Top ranking hyper-parameters on our benchmark slightly"
                " modified to run in reasonable time."
            ),
            parameters={
                "winner_take_all": True,
                "categorical_algorithm": "RANDOM",
                "split_axis": "SPARSE_OBLIQUE",
                "sparse_oblique_normalization": "MIN_MAX",
                "sparse_oblique_num_projections_exponent": 1.0,
            },
        ),
    ]

  @staticmethod
  def capabilities() -> abstract_learner_pb2.LearnerCapabilities:
    return abstract_learner_pb2.LearnerCapabilities(
        support_partial_cache_dataset_format=False
    )
