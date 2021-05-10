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

from tensorflow_decision_forests.keras import core
from yggdrasil_decision_forests.model import abstract_model_pb2  # pylint: disable=unused-import

TaskType = "abstract_model_pb2.Task"  # pylint: disable=invalid-name
AdvancedArguments = core.AdvancedArguments


class RandomForestModel(core.CoreModel):
  r"""Random Forest learning algorithm.

  A Random Forest (https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
  is a collection of deep CART decision trees trained independently and without
  pruning. Each tree is trained on a random subset of the original training
  dataset (sampled with replacement).

  The algorithm is unique in that it is robust to overfitting, even in extreme
  cases e.g. when there is more features than training examples.

  It is probably the most well-known of the Decision Forest training algorithms.

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

  Attributes:
    task: Task to solve (e.g. Task.CLASSIFICATION, Task.REGRESSION,
      Task.RANKING).
    features: Specify the list and semantic of the input features of the model.
      If not specified, all the available features will be used. If specified
      and if `exclude_non_specified_features=True`, only the features in
      `features` will be used by the model. If "preprocessing" is used,
      `features` corresponds to the output of the preprocessing. In this case,
      it is recommended for the preprocessing to return a dictionary of tensors.
    exclude_non_specified_features: If true, only use the features specified in
      `features`.
    preprocessing: Functional model to apply on the input feature before the
      model to train. This preprocessing model can consume and return tensors,
      list of tensors or dictionary of tensors. If specified, the model only
      "sees" the output of the preprocessing (and not the raw input). Can be
      used to prepare the features or to stack multiple models on top of each
      other. Unlike preprocessing done in the tf.dataset, the operation in
      `preprocessing` are serialized with the model.
    ranking_group: Only for `task=Task.RANKING`. Name of a tf.string feature
      that identifies queries in a query/document ranking task. The ranking
      group is not added automatically for the set of features if
      `exclude_non_specified_features=false`.
    temp_directory: Temporary directory used during the training. The space
      required depends on the learner. In many cases, only a temporary copy of a
      model will be there.
    verbose: If true, displays information about the training.
    hyperparameter_template: Override the default value of the hyper-parameters.
      If None (default) the default parameters of the library are used. If set,
      `default_hyperparameter_template` refers to one of the following
      preconfigured hyper-parameter sets. Those sets outperforms the default
      hyper-parameters (either generally or in specific scenarios). You can omit
      the version (e.g. remove "@v5") to use the last version of the template.
      In this case, the hyper-parameter can change in between releases (not
      recommended for training in production).
    advanced_arguments: Advanced control of the model that most users won't need
      to use. See `AdvancedControl` for details.
        - better_default@v1: A configuration that is generally better than the
          default parameters without being more expensive. The parameters are:
          winner_take_all=True.
        - benchmark_rank1@v1: Top ranking hyper-parameters on our benchmark
          slightly modified to run in reasonable time. The parameters are:
          winner_take_all=True, categorical_algorithm="RANDOM",
          split_axis="SPARSE_OBLIQUE", sparse_oblique_normalization="MIN_MAX",
          sparse_oblique_num_projections_exponent=1.0.
    adapt_bootstrap_size_ratio_for_maximum_training_duration: Control how the
      maximum training duration (if set) is applied. If false, the training stop
      when the time is used. If true, adapts the size of the sampled dataset
      used to train each tree such that `num_trees` will train within
      `maximum_training_duration`. Has no effect if there is no maximum training
      duration specified. Default: False.
    allow_na_conditions: If true, the tree training evaluates conditions of the
      type `X is NA` i.e. `X is missing`. Default: False.
    categorical_algorithm: How to learn splits on categorical attributes.
      - `CART`: CART algorithm. Find categorical splits of the form "value \\in
        mask". The solution is exact for binary classification, regression and
        ranking. It is approximated for multi-class classification. This is a
        good first algorithm to use. In case of overfitting (very small dataset,
        large dictionary), the "random" algorithm is a good alternative.
      - `ONE_HOT`: One-hot encoding. Find the optimal categorical split of the
        form "attribute == param". This method is similar (but more efficient)
        than converting converting each possible categorical value into a
        boolean feature. This method is available for comparison purpose and
        generally performs worse than other alternatives.
      - `RANDOM`: Best splits among a set of random candidate. Find the a
        categorical split of the form "value \\in mask" using a random search.
        This solution can be seen as an approximation of the CART algorithm.
        This method is a strong alternative to CART. This algorithm is inspired
        from section "5.1 Categorical Variables" of "Random Forest", 2001.
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
      texts. Minimum number of occurrences of an item to be considered. Default:
      1.
    compute_oob_performances: If true, compute the Out-of-bag evaluation (then
      available in the summary and model inspector). This evaluation is a cheap
      alternative to cross-validation evaluation. Default: True.
    compute_oob_variable_importances: If true, compute the Out-of-bag feature
      importance (then available in the summary and model inspector). Note that
      the OOB feature importance can be expensive to compute. Default: False.
    growing_strategy: How to grow the tree.
      - `LOCAL`: Each node is split independently of the other nodes. In other
        words, as long as a node satisfy the splits "constraints (e.g. maximum
        depth, minimum number of observations), the node will be split. This is
        the "classical" way to grow decision trees.
      - `BEST_FIRST_GLOBAL`: The node with the best loss reduction among all the
        nodes of the tree is selected for splitting. This method is also called
        "best first" or "leaf-wise growth". See "Best-first decision tree
        learning", Shi and "Additive logistic regression : A statistical view of
        boosting", Friedman for more details. Default: "LOCAL".
    in_split_min_examples_check: Whether to check the `min_examples` constraint
      in the split search (i.e. splits leading to one child having less than
      `min_examples` examples are considered invalid) or before the split search
      (i.e. a node can be derived only if it contains more than `min_examples`
      examples). If false, there can be nodes with less than `min_examples`
      training examples. Default: True.
    max_depth: Maximum depth of the tree. `max_depth=1` means that all trees
      will be roots. Negative values are ignored. Default: 16.
    max_num_nodes: Maximum number of nodes in the tree. Set to -1 to disable
      this limit. Only available for `growing_strategy=BEST_FIRST_GLOBAL`.
      Default: None.
    maximum_training_duration_seconds: Maximum training duration of the model
      expressed in seconds. Each learning algorithm is free to use this
      parameter at it sees fit. Enabling maximum training duration makes the
      model training non-deterministic. Default: -1.0.
    min_examples: Minimum number of examples in a node. Default: 5.
    missing_value_policy: Method used to handle missing attribute values.
      - `GLOBAL_IMPUTATION`: Missing attribute values are imputed, with the mean
        (in case of numerical attribute) or the most-frequent-item (in case of
        categorical attribute) computed on the entire dataset (i.e. the
        information contained in the data spec).
      - `LOCAL_IMPUTATION`: Missing attribute values are imputed with the mean
        (numerical attribute) or most-frequent-item (in the case of categorical
        attribute) evaluated on the training examples in the current node.
      - `RANDOM_LOCAL_IMPUTATION`: Missing attribute values are imputed from
        randomly sampled values from the training examples in the current node.
        This method was proposed by Clinic et al. in "Random Survival Forests"
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
    num_trees: Number of individual decision trees. Increasing the number of
      trees can increase the quality of the model at the expense of size,
      training speed, and inference latency. Default: 300.
    sparse_oblique_normalization: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Normalization applied on the features, before
      applying the sparse oblique projections.
      - `NONE`: No normalization.
      - `STANDARD_DEVIATION`: Normalize the feature by the estimated standard
        deviation on the entire train dataset. Also known as Z-Score
        normalization.
      - `MIN_MAX`: Normalize the feature by the range (i.e. max-min) estimated
        on the entire train dataset. Default: None.
    sparse_oblique_num_projections_exponent: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Controls of the number of random projections
      to test at each node as `num_features^num_projections_exponent`. Default:
      None.
    sparse_oblique_projection_density_factor: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Controls of the number of random projections
      to test at each node as `num_features^num_projections_exponent`. Default:
      None.
    split_axis: What structure of split to consider for numerical features.
      - `AXIS_ALIGNED`: Axis aligned splits (i.e. one condition at a time). This
        is the "classical" way to train a tree. Default value.
      - `SPARSE_OBLIQUE`: Sparse oblique splits (i.e. splits one a small number
        of features) from "Sparse Projection Oblique Random Forests", Tomita et
        al., 2020. Default: "AXIS_ALIGNED".
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
      preprocessing: Optional["tf.keras.models.Functional"] = None,
      ranking_group: Optional[str] = None,
      temp_directory: Optional[str] = None,
      verbose: Optional[bool] = True,
      hyperparameter_template: Optional[str] = None,
      advanced_arguments: Optional[AdvancedControl] = None,
      adapt_bootstrap_size_ratio_for_maximum_training_duration: Optional[
          bool] = False,
      allow_na_conditions: Optional[bool] = False,
      categorical_algorithm: Optional[str] = "CART",
      categorical_set_split_greedy_sampling: Optional[float] = 0.1,
      categorical_set_split_max_num_items: Optional[int] = -1,
      categorical_set_split_min_item_frequency: Optional[int] = 1,
      compute_oob_performances: Optional[bool] = True,
      compute_oob_variable_importances: Optional[bool] = False,
      growing_strategy: Optional[str] = "LOCAL",
      in_split_min_examples_check: Optional[bool] = True,
      max_depth: Optional[int] = 16,
      max_num_nodes: Optional[int] = None,
      maximum_training_duration_seconds: Optional[float] = -1.0,
      min_examples: Optional[int] = 5,
      missing_value_policy: Optional[str] = "GLOBAL_IMPUTATION",
      num_candidate_attributes: Optional[int] = 0,
      num_candidate_attributes_ratio: Optional[float] = -1.0,
      num_trees: Optional[int] = 300,
      sparse_oblique_normalization: Optional[str] = None,
      sparse_oblique_num_projections_exponent: Optional[float] = None,
      sparse_oblique_projection_density_factor: Optional[float] = None,
      split_axis: Optional[str] = "AXIS_ALIGNED",
      winner_take_all: Optional[bool] = True,
      explicit_args: Optional[Set[str]] = None):

    learner_params = {
        "adapt_bootstrap_size_ratio_for_maximum_training_duration":
            adapt_bootstrap_size_ratio_for_maximum_training_duration,
        "allow_na_conditions":
            allow_na_conditions,
        "categorical_algorithm":
            categorical_algorithm,
        "categorical_set_split_greedy_sampling":
            categorical_set_split_greedy_sampling,
        "categorical_set_split_max_num_items":
            categorical_set_split_max_num_items,
        "categorical_set_split_min_item_frequency":
            categorical_set_split_min_item_frequency,
        "compute_oob_performances":
            compute_oob_performances,
        "compute_oob_variable_importances":
            compute_oob_variable_importances,
        "growing_strategy":
            growing_strategy,
        "in_split_min_examples_check":
            in_split_min_examples_check,
        "max_depth":
            max_depth,
        "max_num_nodes":
            max_num_nodes,
        "maximum_training_duration_seconds":
            maximum_training_duration_seconds,
        "min_examples":
            min_examples,
        "missing_value_policy":
            missing_value_policy,
        "num_candidate_attributes":
            num_candidate_attributes,
        "num_candidate_attributes_ratio":
            num_candidate_attributes_ratio,
        "num_trees":
            num_trees,
        "sparse_oblique_normalization":
            sparse_oblique_normalization,
        "sparse_oblique_num_projections_exponent":
            sparse_oblique_num_projections_exponent,
        "sparse_oblique_projection_density_factor":
            sparse_oblique_projection_density_factor,
        "split_axis":
            split_axis,
        "winner_take_all":
            winner_take_all,
    }

    if hyperparameter_template is not None:
      learner_params = core._apply_hp_template(
          learner_params, hyperparameter_template,
          self.predefined_hyperparameters(), explicit_args)

    super(RandomForestModel, self).__init__(
        task=task,
        learner="RANDOM_FOREST",
        learner_params=learner_params,
        features=features,
        exclude_non_specified_features=exclude_non_specified_features,
        preprocessing=preprocessing,
        ranking_group=ranking_group,
        temp_directory=temp_directory,
        verbose=verbose,
        advanced_arguments=advanced_arguments)

  @staticmethod
  def predefined_hyperparameters() -> List[core.HyperParameterTemplate]:
    return [
        core.HyperParameterTemplate(
            name="better_default",
            version=1,
            description="A configuration that is generally better than the default parameters without being more expensive.",
            parameters={"winner_take_all": True}),
        core.HyperParameterTemplate(
            name="benchmark_rank1",
            version=1,
            description="Top ranking hyper-parameters on our benchmark slightly modified to run in reasonable time.",
            parameters={
                "winner_take_all": True,
                "categorical_algorithm": "RANDOM",
                "split_axis": "SPARSE_OBLIQUE",
                "sparse_oblique_normalization": "MIN_MAX",
                "sparse_oblique_num_projections_exponent": 1.0
            }),
    ]


class GradientBoostedTreesModel(core.CoreModel):
  r"""Gradient Boosted Trees learning algorithm.

  A GBT (Gradient Boosted [Decision] Tree;
  https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) is a set of shallow decision
  trees trained sequentially. Each tree is trained to predict and then "correct"
  for the errors of the previously trained trees (more precisely each tree
  predict the gradient of the loss relative to the model output).

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

  Attributes:
    task: Task to solve (e.g. Task.CLASSIFICATION, Task.REGRESSION,
      Task.RANKING).
    features: Specify the list and semantic of the input features of the model.
      If not specified, all the available features will be used. If specified
      and if `exclude_non_specified_features=True`, only the features in
      `features` will be used by the model. If "preprocessing" is used,
      `features` corresponds to the output of the preprocessing. In this case,
      it is recommended for the preprocessing to return a dictionary of tensors.
    exclude_non_specified_features: If true, only use the features specified in
      `features`.
    preprocessing: Functional model to apply on the input feature before the
      model to train. This preprocessing model can consume and return tensors,
      list of tensors or dictionary of tensors. If specified, the model only
      "sees" the output of the preprocessing (and not the raw input). Can be
      used to prepare the features or to stack multiple models on top of each
      other. Unlike preprocessing done in the tf.dataset, the operation in
      `preprocessing` are serialized with the model.
    ranking_group: Only for `task=Task.RANKING`. Name of a tf.string feature
      that identifies queries in a query/document ranking task. The ranking
      group is not added automatically for the set of features if
      `exclude_non_specified_features=false`.
    temp_directory: Temporary directory used during the training. The space
      required depends on the learner. In many cases, only a temporary copy of a
      model will be there.
    verbose: If true, displays information about the training.
    hyperparameter_template: Override the default value of the hyper-parameters.
      If None (default) the default parameters of the library are used. If set,
      `default_hyperparameter_template` refers to one of the following
      preconfigured hyper-parameter sets. Those sets outperforms the default
      hyper-parameters (either generally or in specific scenarios). You can omit
      the version (e.g. remove "@v5") to use the last version of the template.
      In this case, the hyper-parameter can change in between releases (not
      recommended for training in production).
    advanced_arguments: Advanced control of the model that most users won't need
      to use. See `AdvancedControl` for details.
        - better_default@v1: A configuration that is generally better than the
          default parameters without being more expensive. The parameters are:
          growing_strategy="BEST_FIRST_GLOBAL".
        - benchmark_rank1@v1: Top ranking hyper-parameters on our benchmark
          slightly modified to run in reasonable time. The parameters are:
          growing_strategy="BEST_FIRST_GLOBAL", categorical_algorithm="RANDOM",
          split_axis="SPARSE_OBLIQUE", sparse_oblique_normalization="MIN_MAX",
          sparse_oblique_num_projections_exponent=1.0.
    adapt_subsample_for_maximum_training_duration: Control how the maximum
      training duration (if set) is applied. If false, the training stop when
      the time is used. If true, the size of the sampled datasets used train
      individual trees are adapted dynamically so that all the trees are trained
      in time. Default: False.
    allow_na_conditions: If true, the tree training evaluates conditions of the
      type `X is NA` i.e. `X is missing`. Default: False.
    categorical_algorithm: How to learn splits on categorical attributes.
      - `CART`: CART algorithm. Find categorical splits of the form "value \\in
        mask". The solution is exact for binary classification, regression and
        ranking. It is approximated for multi-class classification. This is a
        good first algorithm to use. In case of overfitting (very small dataset,
        large dictionary), the "random" algorithm is a good alternative.
      - `ONE_HOT`: One-hot encoding. Find the optimal categorical split of the
        form "attribute == param". This method is similar (but more efficient)
        than converting converting each possible categorical value into a
        boolean feature. This method is available for comparison purpose and
        generally performs worse than other alternatives.
      - `RANDOM`: Best splits among a set of random candidate. Find the a
        categorical split of the form "value \\in mask" using a random search.
        This solution can be seen as an approximation of the CART algorithm.
        This method is a strong alternative to CART. This algorithm is inspired
        from section "5.1 Categorical Variables" of "Random Forest", 2001.
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
      texts. Minimum number of occurrences of an item to be considered. Default:
      1.
    dart_dropout: Dropout rate applied when using the DART i.e. when
      forest_extraction=DART. Default: 0.01.
    forest_extraction: How to construct the forest:
      - MART: For Multiple Additive Regression Trees. The "classical" way to
        build a GBDT i.e. each tree tries to "correct" the mistakes of the
        previous trees.
      - DART: For Dropout Additive Regression Trees. A modification of MART
        proposed in http://proceedings.mlr.press/v38/korlakaivinayak15.pdf.
        Here, each tree tries to "correct" the mistakes of a random subset of
        the previous trees. Default: "MART".
    goss_alpha: Alpha parameter for the GOSS (Gradient-based One-Side Sampling;
      see "LightGBM: A Highly Efficient Gradient Boosting Decision Tree sampling
      method. Default: 0.2.
    goss_beta: Beta parameter for the GOSS (Gradient-based One-Side Sampling)
      sampling method. Default: 0.1.
    growing_strategy: How to grow the tree.
      - `LOCAL`: Each node is split independently of the other nodes. In other
        words, as long as a node satisfy the splits "constraints (e.g. maximum
        depth, minimum number of observations), the node will be split. This is
        the "classical" way to grow decision trees.
      - `BEST_FIRST_GLOBAL`: The node with the best loss reduction among all the
        nodes of the tree is selected for splitting. This method is also called
        "best first" or "leaf-wise growth". See "Best-first decision tree
        learning", Shi and "Additive logistic regression : A statistical view of
        boosting", Friedman for more details. Default: "LOCAL".
    in_split_min_examples_check: Whether to check the `min_examples` constraint
      in the split search (i.e. splits leading to one child having less than
      `min_examples` examples are considered invalid) or before the split search
      (i.e. a node can be derived only if it contains more than `min_examples`
      examples). If false, there can be nodes with less than `min_examples`
      training examples. Default: True.
    l1_regularization: L1 regularization applied to the training loss. Impact
      the tree structures and lead values. Default: 0.0.
    l2_categorical_regularization: L2 regularization applied to the training
      loss for categorical features. Impact the tree structures and lead values.
      Default: 1.0.
    l2_regularization: L2 regularization applied to the training loss for all
      features except the categorical ones. Default: 0.0.
    lambda_loss: Lambda regularization applied to certain training loss
      functions. Only for NDCG loss. Default: 1.0.
    max_depth: Maximum depth of the tree. `max_depth=1` means that all trees
      will be roots. Negative values are ignored. Default: 6.
    max_num_nodes: Maximum number of nodes in the tree. Set to -1 to disable
      this limit. Only available for `growing_strategy=BEST_FIRST_GLOBAL`.
      Default: None.
    maximum_training_duration_seconds: Maximum training duration of the model
      expressed in seconds. Each learning algorithm is free to use this
      parameter at it sees fit. Enabling maximum training duration makes the
      model training non-deterministic. Default: -1.0.
    min_examples: Minimum number of examples in a node. Default: 5.
    missing_value_policy: Method used to handle missing attribute values.
      - `GLOBAL_IMPUTATION`: Missing attribute values are imputed, with the mean
        (in case of numerical attribute) or the most-frequent-item (in case of
        categorical attribute) computed on the entire dataset (i.e. the
        information contained in the data spec).
      - `LOCAL_IMPUTATION`: Missing attribute values are imputed with the mean
        (numerical attribute) or most-frequent-item (in the case of categorical
        attribute) evaluated on the training examples in the current node.
      - `RANDOM_LOCAL_IMPUTATION`: Missing attribute values are imputed from
        randomly sampled values from the training examples in the current node.
        This method was proposed by Clinic et al. in "Random Survival Forests"
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
    sampling_method: Control the sampling of the datasets used to train
      individual trees.
      - NONE: No sampling is applied.
      - RANDOM: Uniform random sampling. Automatically selected if "subsample"
        is set.
      - GOSS: Gradient-based One-Side Sampling. Automatically selected if
        "goss_alpha" or "goss_beta" is set.
      - SELGB: Selective Gradient Boosting. Automatically selected if
        "selective_gradient_boosting_ratio" is set.
        Default: "NONE".
    selective_gradient_boosting_ratio: Ratio of the dataset used to train
      individual tree for the selective Gradient Boosting (Selective Gradient
      Boosting for Effective Learning to Rank; Lucchese et al;
      http://quickrank.isti.cnr.it/selective-data/selective-SIGIR2018.pdf)
      sampling method. Default: 0.01.
    shrinkage: Coefficient applied to each tree prediction. A small value (0.02)
      tends to give more accurate results (assuming enough trees are trained),
      but results in larger models. Analogous to neural network learning rate.
      Default: 0.1.
    sparse_oblique_normalization: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Normalization applied on the features, before
      applying the sparse oblique projections.
      - `NONE`: No normalization.
      - `STANDARD_DEVIATION`: Normalize the feature by the estimated standard
        deviation on the entire train dataset. Also known as Z-Score
        normalization.
      - `MIN_MAX`: Normalize the feature by the range (i.e. max-min) estimated
        on the entire train dataset. Default: None.
    sparse_oblique_num_projections_exponent: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Controls of the number of random projections
      to test at each node as `num_features^num_projections_exponent`. Default:
      None.
    sparse_oblique_projection_density_factor: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Controls of the number of random projections
      to test at each node as `num_features^num_projections_exponent`. Default:
      None.
    split_axis: What structure of split to consider for numerical features.
      - `AXIS_ALIGNED`: Axis aligned splits (i.e. one condition at a time). This
        is the "classical" way to train a tree. Default value.
      - `SPARSE_OBLIQUE`: Sparse oblique splits (i.e. splits one a small number
        of features) from "Sparse Projection Oblique Random Forests", Tomita et
        al., 2020. Default: "AXIS_ALIGNED".
    subsample: Ratio of the dataset (sampling without replacement) used to train
      individual trees for the random sampling method. Default: 1.0.
    use_hessian_gain: Use true, uses a formulation of split gain with a hessian
      term i.e. optimizes the splits to minimize the variance of "gradient /
      hessian. Available for all losses except regression. Default: False.
  """

  @core._list_explicit_arguments
  def __init__(
      self,
      task: Optional[TaskType] = core.Task.CLASSIFICATION,
      features: Optional[List[core.FeatureUsage]] = None,
      exclude_non_specified_features: Optional[bool] = False,
      preprocessing: Optional["tf.keras.models.Functional"] = None,
      ranking_group: Optional[str] = None,
      temp_directory: Optional[str] = None,
      verbose: Optional[bool] = True,
      hyperparameter_template: Optional[str] = None,
      advanced_arguments: Optional[AdvancedControl] = None,
      adapt_subsample_for_maximum_training_duration: Optional[bool] = False,
      allow_na_conditions: Optional[bool] = False,
      categorical_algorithm: Optional[str] = "CART",
      categorical_set_split_greedy_sampling: Optional[float] = 0.1,
      categorical_set_split_max_num_items: Optional[int] = -1,
      categorical_set_split_min_item_frequency: Optional[int] = 1,
      dart_dropout: Optional[float] = 0.01,
      forest_extraction: Optional[str] = "MART",
      goss_alpha: Optional[float] = 0.2,
      goss_beta: Optional[float] = 0.1,
      growing_strategy: Optional[str] = "LOCAL",
      in_split_min_examples_check: Optional[bool] = True,
      l1_regularization: Optional[float] = 0.0,
      l2_categorical_regularization: Optional[float] = 1.0,
      l2_regularization: Optional[float] = 0.0,
      lambda_loss: Optional[float] = 1.0,
      max_depth: Optional[int] = 6,
      max_num_nodes: Optional[int] = None,
      maximum_training_duration_seconds: Optional[float] = -1.0,
      min_examples: Optional[int] = 5,
      missing_value_policy: Optional[str] = "GLOBAL_IMPUTATION",
      num_candidate_attributes: Optional[int] = -1,
      num_candidate_attributes_ratio: Optional[float] = -1.0,
      num_trees: Optional[int] = 300,
      sampling_method: Optional[str] = "NONE",
      selective_gradient_boosting_ratio: Optional[float] = 0.01,
      shrinkage: Optional[float] = 0.1,
      sparse_oblique_normalization: Optional[str] = None,
      sparse_oblique_num_projections_exponent: Optional[float] = None,
      sparse_oblique_projection_density_factor: Optional[float] = None,
      split_axis: Optional[str] = "AXIS_ALIGNED",
      subsample: Optional[float] = 1.0,
      use_hessian_gain: Optional[bool] = False,
      explicit_args: Optional[Set[str]] = None):

    learner_params = {
        "adapt_subsample_for_maximum_training_duration":
            adapt_subsample_for_maximum_training_duration,
        "allow_na_conditions":
            allow_na_conditions,
        "categorical_algorithm":
            categorical_algorithm,
        "categorical_set_split_greedy_sampling":
            categorical_set_split_greedy_sampling,
        "categorical_set_split_max_num_items":
            categorical_set_split_max_num_items,
        "categorical_set_split_min_item_frequency":
            categorical_set_split_min_item_frequency,
        "dart_dropout":
            dart_dropout,
        "forest_extraction":
            forest_extraction,
        "goss_alpha":
            goss_alpha,
        "goss_beta":
            goss_beta,
        "growing_strategy":
            growing_strategy,
        "in_split_min_examples_check":
            in_split_min_examples_check,
        "l1_regularization":
            l1_regularization,
        "l2_categorical_regularization":
            l2_categorical_regularization,
        "l2_regularization":
            l2_regularization,
        "lambda_loss":
            lambda_loss,
        "max_depth":
            max_depth,
        "max_num_nodes":
            max_num_nodes,
        "maximum_training_duration_seconds":
            maximum_training_duration_seconds,
        "min_examples":
            min_examples,
        "missing_value_policy":
            missing_value_policy,
        "num_candidate_attributes":
            num_candidate_attributes,
        "num_candidate_attributes_ratio":
            num_candidate_attributes_ratio,
        "num_trees":
            num_trees,
        "sampling_method":
            sampling_method,
        "selective_gradient_boosting_ratio":
            selective_gradient_boosting_ratio,
        "shrinkage":
            shrinkage,
        "sparse_oblique_normalization":
            sparse_oblique_normalization,
        "sparse_oblique_num_projections_exponent":
            sparse_oblique_num_projections_exponent,
        "sparse_oblique_projection_density_factor":
            sparse_oblique_projection_density_factor,
        "split_axis":
            split_axis,
        "subsample":
            subsample,
        "use_hessian_gain":
            use_hessian_gain,
    }

    if hyperparameter_template is not None:
      learner_params = core._apply_hp_template(
          learner_params, hyperparameter_template,
          self.predefined_hyperparameters(), explicit_args)

    super(GradientBoostedTreesModel, self).__init__(
        task=task,
        learner="GRADIENT_BOOSTED_TREES",
        learner_params=learner_params,
        features=features,
        exclude_non_specified_features=exclude_non_specified_features,
        preprocessing=preprocessing,
        ranking_group=ranking_group,
        temp_directory=temp_directory,
        verbose=verbose,
        advanced_arguments=advanced_arguments)

  @staticmethod
  def predefined_hyperparameters() -> List[core.HyperParameterTemplate]:
    return [
        core.HyperParameterTemplate(
            name="better_default",
            version=1,
            description="A configuration that is generally better than the default parameters without being more expensive.",
            parameters={"growing_strategy": "BEST_FIRST_GLOBAL"}),
        core.HyperParameterTemplate(
            name="benchmark_rank1",
            version=1,
            description="Top ranking hyper-parameters on our benchmark slightly modified to run in reasonable time.",
            parameters={
                "growing_strategy": "BEST_FIRST_GLOBAL",
                "categorical_algorithm": "RANDOM",
                "split_axis": "SPARSE_OBLIQUE",
                "sparse_oblique_normalization": "MIN_MAX",
                "sparse_oblique_num_projections_exponent": 1.0
            }),
    ]


class CartModel(core.CoreModel):
  r"""Cart learning algorithm.

  A CART (Classification and Regression Trees) a decision tree. The non-leaf
  nodes contains conditions (also known as splits) while the leaf nodes contains
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

  Attributes:
    task: Task to solve (e.g. Task.CLASSIFICATION, Task.REGRESSION,
      Task.RANKING).
    features: Specify the list and semantic of the input features of the model.
      If not specified, all the available features will be used. If specified
      and if `exclude_non_specified_features=True`, only the features in
      `features` will be used by the model. If "preprocessing" is used,
      `features` corresponds to the output of the preprocessing. In this case,
      it is recommended for the preprocessing to return a dictionary of tensors.
    exclude_non_specified_features: If true, only use the features specified in
      `features`.
    preprocessing: Functional model to apply on the input feature before the
      model to train. This preprocessing model can consume and return tensors,
      list of tensors or dictionary of tensors. If specified, the model only
      "sees" the output of the preprocessing (and not the raw input). Can be
      used to prepare the features or to stack multiple models on top of each
      other. Unlike preprocessing done in the tf.dataset, the operation in
      `preprocessing` are serialized with the model.
    ranking_group: Only for `task=Task.RANKING`. Name of a tf.string feature
      that identifies queries in a query/document ranking task. The ranking
      group is not added automatically for the set of features if
      `exclude_non_specified_features=false`.
    temp_directory: Temporary directory used during the training. The space
      required depends on the learner. In many cases, only a temporary copy of a
      model will be there.
    verbose: If true, displays information about the training.
    hyperparameter_template: Override the default value of the hyper-parameters.
      If None (default) the default parameters of the library are used. If set,
      `default_hyperparameter_template` refers to one of the following
      preconfigured hyper-parameter sets. Those sets outperforms the default
      hyper-parameters (either generally or in specific scenarios). You can omit
      the version (e.g. remove "@v5") to use the last version of the template.
      In this case, the hyper-parameter can change in between releases (not
      recommended for training in production).
    advanced_arguments: Advanced control of the model that most users won't need
      to use. See `AdvancedControl` for details.
    allow_na_conditions: If true, the tree training evaluates conditions of the
      type `X is NA` i.e. `X is missing`. Default: False.
    categorical_algorithm: How to learn splits on categorical attributes.
      - `CART`: CART algorithm. Find categorical splits of the form "value \\in
        mask". The solution is exact for binary classification, regression and
        ranking. It is approximated for multi-class classification. This is a
        good first algorithm to use. In case of overfitting (very small dataset,
        large dictionary), the "random" algorithm is a good alternative.
      - `ONE_HOT`: One-hot encoding. Find the optimal categorical split of the
        form "attribute == param". This method is similar (but more efficient)
        than converting converting each possible categorical value into a
        boolean feature. This method is available for comparison purpose and
        generally performs worse than other alternatives.
      - `RANDOM`: Best splits among a set of random candidate. Find the a
        categorical split of the form "value \\in mask" using a random search.
        This solution can be seen as an approximation of the CART algorithm.
        This method is a strong alternative to CART. This algorithm is inspired
        from section "5.1 Categorical Variables" of "Random Forest", 2001.
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
      texts. Minimum number of occurrences of an item to be considered. Default:
      1.
    growing_strategy: How to grow the tree.
      - `LOCAL`: Each node is split independently of the other nodes. In other
        words, as long as a node satisfy the splits "constraints (e.g. maximum
        depth, minimum number of observations), the node will be split. This is
        the "classical" way to grow decision trees.
      - `BEST_FIRST_GLOBAL`: The node with the best loss reduction among all the
        nodes of the tree is selected for splitting. This method is also called
        "best first" or "leaf-wise growth". See "Best-first decision tree
        learning", Shi and "Additive logistic regression : A statistical view of
        boosting", Friedman for more details. Default: "LOCAL".
    in_split_min_examples_check: Whether to check the `min_examples` constraint
      in the split search (i.e. splits leading to one child having less than
      `min_examples` examples are considered invalid) or before the split search
      (i.e. a node can be derived only if it contains more than `min_examples`
      examples). If false, there can be nodes with less than `min_examples`
      training examples. Default: True.
    max_depth: Maximum depth of the tree. `max_depth=1` means that all trees
      will be roots. Negative values are ignored. Default: 16.
    max_num_nodes: Maximum number of nodes in the tree. Set to -1 to disable
      this limit. Only available for `growing_strategy=BEST_FIRST_GLOBAL`.
      Default: None.
    maximum_training_duration_seconds: Maximum training duration of the model
      expressed in seconds. Each learning algorithm is free to use this
      parameter at it sees fit. Enabling maximum training duration makes the
      model training non-deterministic. Default: -1.0.
    min_examples: Minimum number of examples in a node. Default: 5.
    missing_value_policy: Method used to handle missing attribute values.
      - `GLOBAL_IMPUTATION`: Missing attribute values are imputed, with the mean
        (in case of numerical attribute) or the most-frequent-item (in case of
        categorical attribute) computed on the entire dataset (i.e. the
        information contained in the data spec).
      - `LOCAL_IMPUTATION`: Missing attribute values are imputed with the mean
        (numerical attribute) or most-frequent-item (in the case of categorical
        attribute) evaluated on the training examples in the current node.
      - `RANDOM_LOCAL_IMPUTATION`: Missing attribute values are imputed from
        randomly sampled values from the training examples in the current node.
        This method was proposed by Clinic et al. in "Random Survival Forests"
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
    sparse_oblique_normalization: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Normalization applied on the features, before
      applying the sparse oblique projections.
      - `NONE`: No normalization.
      - `STANDARD_DEVIATION`: Normalize the feature by the estimated standard
        deviation on the entire train dataset. Also known as Z-Score
        normalization.
      - `MIN_MAX`: Normalize the feature by the range (i.e. max-min) estimated
        on the entire train dataset. Default: None.
    sparse_oblique_num_projections_exponent: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Controls of the number of random projections
      to test at each node as `num_features^num_projections_exponent`. Default:
      None.
    sparse_oblique_projection_density_factor: For sparse oblique splits i.e.
      `split_axis=SPARSE_OBLIQUE`. Controls of the number of random projections
      to test at each node as `num_features^num_projections_exponent`. Default:
      None.
    split_axis: What structure of split to consider for numerical features.
      - `AXIS_ALIGNED`: Axis aligned splits (i.e. one condition at a time). This
        is the "classical" way to train a tree. Default value.
      - `SPARSE_OBLIQUE`: Sparse oblique splits (i.e. splits one a small number
        of features) from "Sparse Projection Oblique Random Forests", Tomita et
        al., 2020. Default: "AXIS_ALIGNED".
    validation_ratio: Ratio of the training dataset used to create the
      validation dataset used to prune the tree. Default: 0.1.
  """

  @core._list_explicit_arguments
  def __init__(self,
               task: Optional[TaskType] = core.Task.CLASSIFICATION,
               features: Optional[List[core.FeatureUsage]] = None,
               exclude_non_specified_features: Optional[bool] = False,
               preprocessing: Optional["tf.keras.models.Functional"] = None,
               ranking_group: Optional[str] = None,
               temp_directory: Optional[str] = None,
               verbose: Optional[bool] = True,
               hyperparameter_template: Optional[str] = None,
               advanced_arguments: Optional[AdvancedControl] = None,
               allow_na_conditions: Optional[bool] = False,
               categorical_algorithm: Optional[str] = "CART",
               categorical_set_split_greedy_sampling: Optional[float] = 0.1,
               categorical_set_split_max_num_items: Optional[int] = -1,
               categorical_set_split_min_item_frequency: Optional[int] = 1,
               growing_strategy: Optional[str] = "LOCAL",
               in_split_min_examples_check: Optional[bool] = True,
               max_depth: Optional[int] = 16,
               max_num_nodes: Optional[int] = None,
               maximum_training_duration_seconds: Optional[float] = -1.0,
               min_examples: Optional[int] = 5,
               missing_value_policy: Optional[str] = "GLOBAL_IMPUTATION",
               num_candidate_attributes: Optional[int] = 0,
               num_candidate_attributes_ratio: Optional[float] = -1.0,
               sparse_oblique_normalization: Optional[str] = None,
               sparse_oblique_num_projections_exponent: Optional[float] = None,
               sparse_oblique_projection_density_factor: Optional[float] = None,
               split_axis: Optional[str] = "AXIS_ALIGNED",
               validation_ratio: Optional[float] = 0.1,
               explicit_args: Optional[Set[str]] = None):

    learner_params = {
        "allow_na_conditions":
            allow_na_conditions,
        "categorical_algorithm":
            categorical_algorithm,
        "categorical_set_split_greedy_sampling":
            categorical_set_split_greedy_sampling,
        "categorical_set_split_max_num_items":
            categorical_set_split_max_num_items,
        "categorical_set_split_min_item_frequency":
            categorical_set_split_min_item_frequency,
        "growing_strategy":
            growing_strategy,
        "in_split_min_examples_check":
            in_split_min_examples_check,
        "max_depth":
            max_depth,
        "max_num_nodes":
            max_num_nodes,
        "maximum_training_duration_seconds":
            maximum_training_duration_seconds,
        "min_examples":
            min_examples,
        "missing_value_policy":
            missing_value_policy,
        "num_candidate_attributes":
            num_candidate_attributes,
        "num_candidate_attributes_ratio":
            num_candidate_attributes_ratio,
        "sparse_oblique_normalization":
            sparse_oblique_normalization,
        "sparse_oblique_num_projections_exponent":
            sparse_oblique_num_projections_exponent,
        "sparse_oblique_projection_density_factor":
            sparse_oblique_projection_density_factor,
        "split_axis":
            split_axis,
        "validation_ratio":
            validation_ratio,
    }

    if hyperparameter_template is not None:
      learner_params = core._apply_hp_template(
          learner_params, hyperparameter_template,
          self.predefined_hyperparameters(), explicit_args)

    super(CartModel, self).__init__(
        task=task,
        learner="CART",
        learner_params=learner_params,
        features=features,
        exclude_non_specified_features=exclude_non_specified_features,
        preprocessing=preprocessing,
        ranking_group=ranking_group,
        temp_directory=temp_directory,
        verbose=verbose,
        advanced_arguments=advanced_arguments)

  @staticmethod
  def predefined_hyperparameters() -> List[core.HyperParameterTemplate]:
    return []
