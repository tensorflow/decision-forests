# Migrating from Neural Networks

**TensorFlow Decision Forests** (**TF-DF**) is a collection of Decision Forest
(**DF**) algorithms available in TensorFlow. Decision Forests work differently
than Neural Networks (**NN**): DFs generally do not train with
backpropagation, or in mini-batches. Therefore, TF-DF pipelines have a few
differences from other TensorFlow pipelines.

This document is a list of those differences, and a guide to updating TF
pipelines to use TF-DF

This doc assumes familiarity with the
[beginner colab](tutorials/beginner_colab.ipynb).

## Table of Contents

<!--ts-->

*   [Migrating from Neural Networks](#migrating-from-neural-networks)
    *   [Table of Contents](#table-of-contents)
    *   [Dataset and Features](#dataset-and-features)
        *   [Validation dataset](#validation-dataset)
        *   [Dataset I/O](#dataset-io)
            *   [Train for exactly 1 epoch](#train-for-exactly-1-epoch)
            *   [Do not shuffle the dataset](#do-not-shuffle-the-dataset)
            *   [Do not tune the batch size](#do-not-tune-the-batch-size)
        *   [Large Datasets](#large-datasets)
            *   [How many examples to use](#how-many-examples-to-use)
        *   [Feature Normalization / Preprocessing](#feature-normalization--preprocessing)
            *   [Do not transform data with feature columns](#do-not-transform-data-with-feature-columns)
            *   [Do not preprocess the features](#do-not-preprocess-the-features)
            *   [Do not normalize numerical features](#do-not-normalize-numerical-features)
            *   [Do not encode categorical features (e.g. hashing, one-hot, or
                embedding)](#do-not-encode-categorical-features-eg-hashing-one-hot-or-embedding)
            *   [How to handle text features](#how-to-handle-text-features)
            *   [Do not replace missing features by magic values](#do-not-replace-missing-features-by-magic-values)
            *   [Handling Images and Time series](#handling-images-and-time-series)
    *   [Training Pipeline](#training-pipeline)
        *   [Don't use hardware accelerators e.g. GPU, TPU](#dont-use-hardware-accelerators-eg-gpu-tpu)
        *   [Don't use checkpointing or mid-training hooks](#dont-use-checkpointing-or-mid-training-hooks)
        *   [Model Determinism](#model-determinism)
        *   [Training Configuration](#training-configuration)
            *   [Specify a task (e.g. classification, ranking) instead of a loss
                (e.g. binary
                cross-entropy)](#specify-a-task-eg-classification-ranking-instead-of-a-loss-eg-binary-cross-entropy)
            *   [Hyper-parameters are semantically stable](#hyper-parameters-are-semantically-stable)
        *   [Model debugging](#model-debugging)
            *   [Simple model summary](#simple-model-summary)
            *   [Training Logs and Tensorboard](#training-logs-and-tensorboard)
            *   [Feature importance](#feature-importance)
            *   [Plotting the trees](#plotting-the-trees)
            *   [Access the tree structure](#access-the-tree-structure)
            *   [Do not use TensorFlow distribution strategies](#do-not-use-tensorflow-distribution-strategies)
            *   [Stacking Models](#stacking-models)
            *   [Migrating from tf.estimator.BoostedTrees
                {Classifier/Regressor/Estimator}](#migrating-from-tfestimatorboostedtrees-classifierregressorestimator)
    *   [For Yggdrasil users](#for-yggdrasil-users)

<!-- Added by: gbm, at: Mon 10 May 2021 03:50:43 PM CEST -->

<!--te-->

## Dataset and Features

### Validation dataset

Unlike the standard Neural Network training paradigm, TF-DF models do not need a
validation dataset to monitor overfitting, or to stop training early. If you
already have a train/validation/test split, and you are using the validation for
one of those reasons, it is safe to train your TF-DF on train+validation (unless
the validation split is also used for something else, like hyperparameter
tuning).

```diff {.bad}
- model.fit(train_ds, validation_data=val_ds)
```

```diff {.good}
+ model.fit(train_ds.concatenate(val_ds))

# Or just don't create a validation dataset
```

**Rationale:** The TF-DF framework is composed of multiple algorithms. Some of
them do not use a validation dataset (e.g. Random Forest) while some others do
(e.g. Gradient Boosted Trees). Algorithms that do might benefit from different
types and size of validation datasets. Therefore, if a validation dataset is
needed, it will be extracted automatically from the training dataset.

### Dataset I/O

#### Train for exactly 1 epoch

```diff {.bad}
# Number of epochs in Keras
- model.fit(train_ds, num_epochs=5)

# Number of epochs in the dataset
- train_ds = train_ds.repeat(5)
- model.fit(train_ds)
```

```diff {.good}
+ model.fit(train_ds)
```

**Rationale:** Users of neural networks often train a model for N steps (which
may involve looping over the dataset > 1 time), because of the nature of
[SGD](https://en.wikipedia.org/wiki/Stochastic_gradient_descent). TF-DF trains
by reading the whole dataset and then running the training at the end. 1 epoch
is needed to read the full dataset, and any extra steps will result in
unnecessary data I/O, as well as slower training.

#### Do not shuffle the dataset

Datasets do not need to be shuffled (unless the input_fn is reading only a
sample of the dataset).

```diff {.bad}
- train_ds = train_ds.shuffle(5)
- model.fit(train_ds)
```

```diff {.good}
+ model.fit(train_ds)
```

**Rationale:** TF-DF shuffles access to the data internally after reading the
full dataset into memory. TF-DF algorithms are deterministic (if the user does
not change the random seed). Enabling shuffling will only make the algorithm non
deterministic. Shuffling does make sense if the input dataset is ordered and the
input_fn is only going to read a sample of it (the sample should be random).
However, this will make the training procedure non-deterministic.

#### Do not tune the batch size

The batch size will not affect the model quality

```diff {.bad}
- train_ds = train_ds.batch(hyper_parameter_batch_size())
- model.fit(train_ds)
```

```diff {.good}
# The batch size does not matter.
+ train_ds = train_ds.batch(64)
+ model.fit(train_ds)
```

***Rationale:*** Since TF-DF is always trained on the full dataset after it is
read, the model quality will not vary based on the batch size (unlike mini-batch
training algorithms like SGD where parameters like learning rate need to be
tuned jointly). Thus it should be removed from hyperparameter sweeps. The batch
size will only have an impact on the speed of dataset I/O.

### Large Datasets

Unlike neural networks, which can loop over mini-batches of a large dataset
infinitely, decision forests require a finite dataset that fits in memory for
their training procedures. The size of the dataset has performance and memory
implications.

There are diminishing returns for increasing the size of the dataset, and DF
algorithms arguably need fewer examples for convergence than large NN models.
Instead of scaling the number of training steps (as in a NN), you can try
scaling the amount of data to see where the compute tradeoff makes sense.
Therefore **it is a good idea to first try training on a (small) subset of the
dataset.**

#### How many examples to use

**It should fit in memory on the machine the model is training on**:

*   Note that this is not the same as the size of the examples on disk.

*   As a rule of thumb one numerical or categorical value uses 4 bytes of
    memory. So, a dataset with 100 features and 25 million examples will take
    ~10GB (= 100 * 25 *10^6 * 4 bytes) of memory.

*   Categorical-set features (e.g. tokenized text) take more memory (4 bytes per
    token + 12 bytes per features).

**Consider your training time budget**

*   While generally faster than NN for smaller datasets (e.g. <100k examples),
    DF training algorithms do not scale linearly with the dataset size; rather,
    ~O(features x num_examples x log(num_examples)) in most cases.

*   The training time depends on the hyper-parameters. The most impactful
    parameters are: (1) the number of trees (`num_trees`), (2) the example
    sampling rate (`subsample` for GBT), and (3) the attribute sampling rate
    (`num_candidate_attributes_ratio`)

*   Categorical-set features are more expensive that other features. The cost is
    controlled by the `categorical_set_split_greedy_sampling` parameter.

*   Sparse Oblique features (disabled by default) give good results but are
    expensive to compute.

**Rules of thumb for scaling up data**

We suggest starting with a small slice of the data (<10k examples), which should
allow you to train a TF-DF model in seconds or a few minutes in most cases. Then
you can increase the data at a fixed rate (e.g. 40% more each time), stopping
when validation set performance does not improve or the dataset no longer fits
in memory.

### Feature Normalization / Preprocessing

#### Do not transform data with feature columns

TF-DF models do not require explicitly providing feature semantics and
transformations. By default, all of the features in the dataset (other than the
label) will be detected and used by the model. The feature semantics will be
auto-detected, and can be overridden manually if needed.

```diff {.bad}
# Estimator code
- feature_columns = [
-   tf.feature_column.numeric_column(feature_1),
-   tf.feature_column.categorical_column_with_vocabulary_list(feature_2, ['First', 'Second', 'Third'])
-   ]
- model = tf.estimator.LinearClassifier(feature_columns=feature_columnes)
```

```diff {.good}
# Use all the available features. Detect the type automatically.
+ model = tfdf.keras.GradientBoostedTreesModel()
```

You can also specify a subset of input features:

```diff {.good}
+ features = [
+   tfdf.keras.FeatureUsage(name="feature_1"),
+   tfdf.keras.FeatureUsage(name="feature_2")
+   ]
+ model = tfdf.keras.GradientBoostedTreesModel(features=features, exclude_non_specified_features=True)
```

If necessary, you can force the semantic of a feature.

```diff {.good}
+ forced_features = [
+   tfdf.keras.FeatureUsage(name="feature_1", semantic=tfdf.keras.FeatureSemantic.CATEGORICAL),
+   ]
+ model = tfdf.keras.GradientBoostedTreesModel(features=features)
```

**Rationale:** While certain models (like Neural Networks) require a
standardized input layer (e.g. mappings from different feature types →
embeddings), TF-DF models can consume categorical and numerical features
natively, as well as auto-detect the semantic types of the features based on the
data.

#### Do not preprocess the features

Decision tree algorithms do not benefit from some of the classical feature
preprocessing used for Neural Networks. Below, some of the more common feature
processing strategies are explicitly listed, but a safe starting point is to
remove all pre-processing that was designed to help neural network training.

#### Do not normalize numerical features

```diff {.bad}
- def zscore(value):
-   return (value-mean) / sd

- feature_columns = [tf.feature_column.numeric_column("feature_1",normalizer_fn=zscore)]
```

**Rational:** Decision forest algorithms natively support non-normalized
numerical features, since the splitting algorithms do not do any numerical
transformation of the input. Some types of normalization (e.g. zscore
normalization) will not help numerical stability of the training procedure, and
some (e.g. outlier clipping) may hurt the expressiveness of the final model.

#### Do not encode categorical features (e.g. hashing, one-hot, or embedding)

```diff {.bad}
- integerized_column = tf.feature_column.categorical_column_with_hash_bucket("feature_1",hash_bucket_size=100)
- feature_columns = [tf.feature_column.indicator_column(integerized_column)]
```

```diff {.bad}
- integerized_column = tf.feature_column.categorical_column_with_vocabulary_list('feature_1', ['bob', 'george', 'wanda'])
- feature_columns = [tf.feature_column.indicator_column(integerized_column)]
```

**Rationale:** TF-DF has native support for categorical features, and will treat
a “transformed” vocabulary item as just another item in its internal vocabulary
(which can be configured via model hyperparameters). Some transformations (like
hashing) can be lossy. Embeddings are not supported unless they are pre-trained,
since Decision Forest models are not differentiable (see
[intermediate colab](tutorials/intermediate_colab.ipynb)). Note that
domain-specific vocabulary strategies (e.g. stopword removal, text
normalization) may still be helpful.

#### How to handle text features

TF-DF supports [categorical-set features](https://arxiv.org/abs/2009.09991)
natively. Therefore, bags of tokenized n-grams can be consumed natively.

Alternatively, text can also be consumed through
[a pre-trained embedding](#handling-images-and-time-series).

Categorical-sets are sample efficient on small datasets, but expensive to train
on large datasets. Combining categorical-sets and a pre-trained embedding can
often yield better results than if either is used alone.

#### Do not replace missing features by magic values

**Rationale:** TF-DF has native support for missing values. Unlike neural
networks, which may propagate NaNs to the gradients if there are NaNs in the
input, TF-DF will train optimally if the algorithm sees the difference between
missing and a sentinel value.

```diff {.bad}
- feature_columns = [
- tf.feature_column.numeric_column("feature_1", default_value=0),
- tf.feature_column.numeric_column("feature_1_is_missing"),
- ]
```

#### Handling Images and Time series

There is no standard algorithm for consuming image or time series features in
Decision Forests, so some extra work is required to use them.

**Rationale:** Convolution, LSTM, attention and other sequence processing
algorithms are neural network specific architectures.

It is possible to handle these features using the following strategies:

*   Feature Engineering

    *   Images: Using image with Random Forest was popular at some point (e.g.

        [Microsoft Kinect](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/BodyPartRecognition.pdf),
        but today, neural nets are state of the art.

    *   Time series:
        [[Moving statistics](http://framework.mathieu.guillame-bert.com/documentation_honey_tutorial_beginner)]
        can work surprisingly well for time series data that has relatively few
        examples (e.g. vital signs in medical domain).

    *   Embedding modules: Neural network embedding modules can provide rich
        features for a decision forest algorithm. The
        [intermediate colab](tutorials/intermediate_colab.ipynb) shows how to
        combine a tf-hub embedding and a TF-DF model.

## Training Pipeline

### Don't use hardware accelerators e.g. GPU, TPU

TF-DF training does not (yet) support hardware accelerators. All training and
inference is done on the CPU (sometimes using SIMD).

Note that TF-DF inference on CPU (especially when served using Yggdrasil C++
libraries) can be surprisingly fast (sub-microsecond per example per cpu core).

### Don't use checkpointing or mid-training hooks

TF-DF does not (currently) support model checkpointing, meaning that hooks
expecting the model to be usable before training is completed are largely
unsupported. The model will only be available after it trains the requested
number of trees (or stops early).

Keras hooks relying on the training step will also not work – due to the nature
of TF-DF training, the model trains at the end of the first epoch, and will be
constant after that epoch. The step only corresponds to the dataset I/O.

### Model Determinism

The TF-DF training algorithm is deterministic, i.e. training twice on the same
dataset will give the exact same model. This is different from neural networks
trained with TensorFlow. To preserve this determinism, users should ensure that
dataset reads are deterministic as well.

### Training Configuration

#### Specify a task (e.g. classification, ranking) instead of a loss (e.g. binary cross-entropy)

```diff {.bad}
- model = tf.keras.Sequential()
- model.add(Dense(64, activation=relu))
- model.add(Dense(1)) # One output for binary classification

- model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
-               optimizer='adam',
-               metrics=['accuracy'])
```

```diff {.good}
# The loss is automatically determined from the task.
+ model = tfdf.keras.GradientBoostedTreesModel(task=tf.keras.Task.CLASSIFICATION)

# Optional if you want to report the accuracy.
+ model.compile(metrics=['accuracy'])
```

**Rationale:** Not all TF-DF learning algorithms use a loss. For those that do,
the loss is automatically detected from the task and printed in the model
summary. You can also override it with the loss hyper-parameter.

#### Hyper-parameters are semantically stable

All the hyper-parameters have default values. Those values are reasonable first
candidates to try. Default hyper-parameter values are guaranteed to never
change. For this reason, new hyper-parameters or algorithm improvements are
disabled by default.

Users that wish to use the latest algorithms, but who do not want to optimize
the hyper-parameters themself can use the "hyper-parameter templates" provided
by TF-DF. New hyperparameter templates will be released with updates to the
package.

```python {.good}
# Model with default hyper-parameters.
model = tfdf.keras.GradientBoostedTreesModel()

# List the hyper-parameters (with default value) and hyper-parameters templates of the GBT learning algorithm (in colab)
?tfdf.keras.GradientBoostedTreesModel

# Use a hyper-parameter template.
model = tfdf.keras.GradientBoostedTreesModel(hp_template="winner_1")

# Change one of the hyper-parameters.
model = tfdf.keras.GradientBoostedTreesModel(num_trees=500)

# List all the learning algorithms available
tfdf.keras.get_all_models()
```

### Model debugging

This section presents some way you can look/debug/interpret the model. The
[beginner colab](tutorials/beginner_colab.ipynb) contains an end-to-end example.

#### Simple model summary

```python
# Text description of the model, training logs, feature importances, etc.
model.summary()
```

#### Training Logs and Tensorboard

Note: Unlike in NNs, training logs are not available in tensorboard until after
training is complete.

```python
# List of metrics
logs = model.make_inspector().training_logs()
print(logs)
```

Or using TensorBoard:

```python
%load_ext tensorboard
model.make_inspector().export_to_tensorboard("/tmp/tensorboard_logs")
%tensorboard --logdir "/tmp/tensorboard_logs"
```

#### Feature importance

```python
model.make_inspector().variable_importances()
```

#### Plotting the trees

```python
tfdf.model_plotter.plot_model_in_colab(model, tree_idx=0)
```

#### Access the tree structure

```python
tree = model.make_inspector().extract_tree(tree_idx=0)
print(tree)
```

(See [advanced colab](tutorials/advanced_colab.ipynb))

#### Do not use TensorFlow distribution strategies

TF-DF does not yet support TF distribution strategies. Multi-worker setups will
be ignored, and the training will only happen on the manager.

```diff {.bad}
- with tf.distribute.MirroredStrategy():
-    model = ...
```

```diff {.good}
+ model = ....
```

#### Stacking Models

TF-DF models do not backpropagate gradients. As a result, they cannot be
composed with NN models unless the NNs are already trained.

#### Migrating from tf.estimator.BoostedTrees {Classifier/Regressor/Estimator}

Despite sounding similar, The TF-DF and Estimator boosted trees are different
algorithms. TF-DF implements the classical
[Random Forest](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) and
[Gradient Boosted Machine (using Trees)](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)
papers. The tf.estimator.BoostedTreesEstimator is an approximate Gradient
Boosted Trees algorithm with a mini-batch training procedure described in
[this paper](http://ecmlpkdd2017.ijs.si/papers/paperID705.pdf)

Some hyper-parameters have similar semantics (e.g. num_trees), but they have
different quality implications. If you tuned the hyperparameters on your
tf.estimator.BoostedTreesEstimator, you will need to re-tune your
hyperparameters within TF-DF to obtain the optimal results.

## For Yggdrasil users

[Yggdrasil Decision Forest](https://github.com/google/yggdrasil-decision-forests)
is the core training and inference library used by TF-DF. Training configuration
and models are cross-compatible (i.e. models trained with TF-DF can be used with
Yggdrasil inference).

However, some of the Yggdrasil algorithms are not (yet) available in TF-DF.

-   Gradient Boosted Tree with sharded sampling.
