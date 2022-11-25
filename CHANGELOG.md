# Changelog

## HEAD

### Features

-   Setting "subsample" is enough enable random subsampling (to need to also set
    "sampling_method=RANDOM").

## 1.1.0 - 2022-11-18

### Features

-   Support for Tensorflow Serving APIs.
-   Add support for zipped Yggdrasil Decision Forests model for
    `yggdrasil_model_to_keras_model`.
-   Added model prediction tutorial.
-   Prevent premature stopping of GBT training through new parameter
    `early_stopping_initial_iteration`.

### Fix

-   Using loaded datasets with TF-DF no longer fails (Github #131).
-   Automatically infer the semantic of int8 values as numerical (was
    categorical before).
-   Build script fixed
-   Model saving no longer fails when using invalid feature names.
-   Added keyword to pandas dataset drop (Github #135).

## 1.0.1 - 2022-09-07

### Fix

-   Issue in the application of auditwheel in TF 1.0.0.

## 1.0.0 - 2022-09-07

### Features

-   Add customization of the number of IO threads when using
    `fit_on_dataset_path`.

### Fix

-   Improved documentation
-   Improved testing and stability

## 0.2.7 - 2022-06-15

### Features

-   Multithreading of the oblique splitter for gradient boosted tree models.
-   Support for pure serving model i.e. model containing only serving data.
-   Add "edit_model" cli tool.

### Fix

-   Remove bias toward low outcome in uplift modeling.

## 0.2.6 - 2022-05-17

### Features

-   Support for TensorFlow 2.9.1

## 0.2.5 - 2022-05-17

### Features

-   Adds the `contrib` module for contributed, non-core functionality.
-   Adds `contrib.scikit_learn_model_converter`, which facilitates converting
    Scikit-Learn tree-based models into TF-DF models.
-   Discard hessian splits with score lower than the parents. This change has
    little effect on the model quality, but it can reduce its size.
-   Add internal flag `hessian_split_score_subtract_parent` to subtract the
    parent score in the computation of an hessian split score.
-   Add support for hyper-parameter optimizers (also called tuner).
-   Add text pretty print of trees with `tree.pretty()` or `str(tree)`.
-   Add support for loading YDF models with file prefixes. Newly created models
    have a random prefix attached to them. This allows combining multiple models
    in Keras.
-   Add support for discretized numerical features.

## 0.2.4 - 2021-02-04

### Features

-   Support for TensorFlow 2.8.

## 0.2.3 - 2021-01-27

### Features

-   Honest Random Forests (also work with Gradient Boosted Tree and CART).
-   Can train Random Forests with example sampling without replacement.
-   Add support for Focal Loss with Gradient Boosted Trees.
-   Add support for MacOS.

### Fixes

-   Incorrect default evaluation of categorical split with uplift tasks. This
    was making uplift models with missing categorical values perform worst, and
    made the inference of uplift model possibly slower.
-   Fix `pd_dataframe_to_tf_dataset` on Pandas dataframe not containing arrays.

## 0.2.2 - 2021-12-13

### Features

-   Surface the `validation_interval_in_trees`,
    `keep_non_leaf_label_distribution` and 'random_seed' hyper-parameters.
-   Add the `batch_size` argument in the `pd_dataframe_to_tf_dataset` utility.
-   Automatically determine the number of threads if `num_threads=None`.
-   Add constructor argument `try_resume_training` to facilitate resuming
    training.
-   Check that the training dataset is well configured for TF-DF e.g. no repeat
    operation, has a large enough batch size, etc. The check can be disabled
    with `check_dataset=False`.
-   When a model is created manually with the model builder, and if the dataspec
    is not provided, tries to adapt the dataspec so that the model looks as if
    it was trained with the global imputation strategy for missing values (i.e.
    missing_value_policy: GLOBAL_IMPUTATION). This makes manually created models
    more likely to be compatible with the fast inference engines.
-   TF-DF models `fit` method now passes the `validation_data` to the Yggdrasil
    learners. This is used for example for early stopping in the case of GBT
    model.
-   Add the "loss" parameter of the GBT model directly in the model constructor.
-   Control the amount of training logs displayed in the notebook (if using
    notebook) or in the console with the `verbose` constructor argument and
    `fit` parameter of the model.

### Fixes

-   `num_candidate_attributes` is not ignored anymore when
    `num_candidate_attributes_ratio=-1`.
-   Use the median bucket split value strategy in the discretized numerical
    splitters (local and distributed).
-   Surface the `max_num_scanned_rows_to_accumulate_statistics` parameter to
    control how many examples are scanned to determine the feature statistics
    when training from a file dataset with `fit_on_dataset_path`.

## 0.2.1 - 2021-11-05

### Features

-   Compatibility with TensorFlow 2.7.0.

## 0.2.0 - 2021-10-29

### Features

-   Add advanced option `predict_single_probability_for_binary_classification`
    to generate prediction tensors of shape [batch_size, 2] for binary
    classification model.
-   Add support for weighted training.
-   Add support for permutation variable importance in the GBT learner with the
    `compute_permutation_variable_importance` parameter.
-   Support for tf.int8 and tf.int16 values.
-   Support for distributed gradient boosted trees learning. Currently, the TF
    ParameterServerStrategy distribution strategy is only available in
    monolithic TF-DF builds. The Yggdrasil Decision Forest GRPC distribute
    strategy can be used instead.
-   Support for training from dataset stored on disk in CSV and RecordIO format
    (instead of creating a tensorflow dataset). This option is currently more
    efficient for distributed training (until the ParameterServerStrategy
    support per-worker datasets).
-   Add `max_vocab_count` argument to the model constructor. The existing
    `max_vocab_count` argument in `FeatureUsage` objects take precedence.

### Fixes

-   Missing filtering of unique values in the categorical-set training feature
    accumulator. Was responsible for a small (e.g. ~0.5% on SST2 dataset) drop
    of accuracy compared to the C++ API.
-   Fix broken support for `max_vocab_count` in a `FeatureUsage` with type
    `CATEGORICAL_SET`.

## 0.1.9 - 2021-08-31

### Features

-   Disable tree pruning in the CART algorithm if the validation dataset is
    empty (i.e. `validation_ratio=0`).
-   Migration to Tensorflow 2.6. You will see an `undefined symbol` error if you
    install this version with a TensorFlow version different than 2.6. Previous
    versions were compiled for TF 2.5.

### Fixes

-   Fix failure from
    [Github Issue #45](https://github.com/tensorflow/decision-forests/issues/45)
    where the wrong field was accessed for leaf node distributions.
-   Fix saving of categorical features specification in the Builder.

## 0.1.8 - 2021-07-28

### Features

-   Model can be composed with the functional Keras API before being trained.
-   Makes all the Yggdrasil structural variable importances available.
-   Makes getting the variable importance instantaneous.
-   Surface the `name` argument in the model classes constructors.
-   Add a `postprocessing` model constructor argument to easy apply
    post-processing on the model predictions without relying on the Keras
    Functional API.
-   Add `extract_all_trees` method in the model inspector to efficiently exact
    all the trees.
-   Add `num_threads` constructor argument to control the number of training
    threads without using the advanced configuration.
-   By default, remove the temporary directory used to train the model when the
    model python object is garbage collected.
-   Add the `import_dataspec` constructor argument to the model builder to
    import the feature definition and dictionaries (instead of relying on
    automatic discovery).

### Changes

-   When saving a model in a directory already containing a model, only the
    `assets` directory is entirely removed before the export (instead of the
    entire model directory).

### Fixes

-   Wrong label shape in the model inspector's objective field for
    pre-integerized labels.

## 0.1.7 - 2021-06-23

### Features

-   Add more of characters to the non-recommended list of feature name
    characters.
-   Make the inference op multi-thread compatible.
-   Print an explicit error and some instructions when training a model with a
    Pandas dataframe.
-   `pd_dataframe_to_tf_dataset` can automatically rename feature to make them
    compatible with SavedModel export signatures.
-   `model.save(...)` can override an existing model.
-   The link function of GBT model can be removed. For example, a binary
    classification GBT model trained with apply_link_function=False will output
    logits.

## 0.1.6 - 2021-06-07

### Features

-   Add hyper-parameter `sorting_strategy` to disable the computation of the
    pre-sorted index (slower to train, but consumes less memory).
-   Format wrapper code for colab help display.
-   Raises an error when a feature name is not compatible (e.g. contains a
    space).

## 0.1.5 - 2021-05-26

### Features

-   Raise an error of the number of classes is greater than 100 (can be
    disabled).
-   Raise an error if the model's task does not match the
    `pd_dataframe_to_tf_dataset`'s task.

### Bug fix

-   Fix failure when input feature contains commas.

## 0.1.4 - 2021-05-21

### Features

-   Stop the training when interrupting a colab cell / typing ctrl-c.
-   `model.fit` support training callbacks and a validation dataset.

### Bug fix

-   Fix failure when there are not input features.

## 0.1.3 - 2021-05-19

### Features

-   Register new inference engines.

## 0.1.2 - 2021-05-18

### Features

-   Inference engines: QuickScorer Extended and Pred

## 0.1.1 - 2021-05-17

### Features

-   Migration to TensorFlow 2.5.0.
-   By default, use a pre-compiled version of the OP wrappers.

### Bug fix

-   Add missing `plotter.js` from Pip package.
-   Use GitHub version of Yggdrasil Decision Forests by default.

## 0.1.0 - 2021-05-11

Initial Release of TensorFlow Decision Forests.

### Features

-   Random Forest learner.
-   Gradient Boosted Tree learner.
-   CART learner.
-   Model inspector: Inspect the internal model structure.
-   Model plotter: Plot decision trees.
-   Model builder: Create model "by hand".
