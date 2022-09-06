# Distributed Training

<!-- docs_infra:strip_begin -->

## Table of Contents

<!--ts-->

<!--te-->

<!-- docs_infra:strip_end -->

**Warning:** (Currently) Distributed training with TensorFlow Parameter Server
is not available with Pip package version of TensorFlow Decision Forests.
Instead of the *TensorFlow Parameter Server*, use the
[Yggdrasil Decision Forest worker](https://www.tensorflow.org/decision_forests/distributed_training#using_yggdrasil_decision_forest_for_both_dataset_reading_and_model_training)
option. All versions of distributed training are available with monolithic
TensorFlow build (e.g., internal build).

## Introduction

Distributed training makes it possible to train models quickly on large
datasets. Not all models support distributed training. Hyper-parameter tuning
always benefit from distributed training.

See the
[distributed training](https://github.com/google/yggdrasil-decision-forests/blob/main/documentation/user_manual.md#distributed-training)
section in the Yggdrasil Decision Forests user manual for details about the
available distributed training algorithms. When using distributed training with
TF Parameter Server in TF-DF, Yggdrasil Decision Forests is effectively running
the `TF_DIST` distribute implementation.

While the learning algorithms remain the same, TF-DF supports three way to
execute distributed training:

1.  [Simplest option] Using Yggdrasil Decision Forest for dataset reading and TF
    Parameter Server for model training.
1.  [The most TensorFlow like option] Using TF Parameter Server for both dataset
    reading and model training.
1.  Using Yggdrasil Decision Forest for both dataset reading and model training.

**Limitations:**

-   Currently (May. 2022), the version of TF-DF distributed on PyPi does not
    support distributed training with the TF Parameter Server distribution
    strategy. In this case, use the Yggdrasil Decision Forest for both dataset
    reading and model training i.e. use the GRPC distribute strategy.
-   Using Yggdrasil Decision Forest for dataset reading does not support
    TensorFlow preprocessing.

## Examples

Following are some examples of distributed training.

### [Simplest option] Using Yggdrasil Decision Forest for dataset reading and TF Parameter Server for model training.

Start a set of
[Parameter Server Strategy workers](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/ParameterServerStrategy).
Then:

```python
import tensorflow_decision_forests as tfdf
import tensorflow as tf

strategy = tf.distribute.experimental.ParameterServerStrategy(...)

with strategy.scope():
  model = tfdf.keras.DistributedGradientBoostedTreesModel()

model.fit_on_dataset_path(
    train_path="/path/to/dataset@100000",
    label_key="label_key",
    dataset_format="tfrecord+tfe")

print("Trained model")
model.summary()
```

See Yggdrasil Decision Forests
[supported formats](https://github.com/google/yggdrasil-decision-forests/blob/main/documentation/user_manual.md#dataset-path-and-format)
for the possible values of `dataset_format`.

### [The most TensorFlow like option] Using TF Parameter Server for both dataset reading and model training

Start a set of
[Parameter Server Strategy workers](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/ParameterServerStrategy).
Then:

```python
import tensorflow_decision_forests as tfdf
import tensorflow as tf

def dataset_fn(context, paths):

  # Like for non-distributed training, each example should be visited exactly
  # once during the training. In addition, for optimal training speed, the
  # reading of the examples should be distributed among the workers (instead
  # of being read by a single worker, or read and discarded multiple times).
  #
  # In other words, don't add a "repeat" statement and make sure to shard the
  # dataset at the file level and not at the example level.

  ds_path = tf.data.Dataset.from_tensor_slices(paths)

  if context is not None:
    # Split the dataset among the workers.
    # Note: You cannot use 'context.num_input_pipelines' with ParameterServerV2.
    current_worker = tfdf.keras.get_worker_idx_and_num_workers(context)
    ds_path = ds_path.shard(
        num_shards=current_worker.num_workers,
        index=current_worker.worker_idx)

  def read_csv_file(path):
    numerical = tf.constant([0.0], dtype=tf.float32)
    categorical_string = tf.constant(["NA"], dtype=tf.string)
    csv_columns = [
        numerical,  # feature 1
        categorical_string,  # feature 2
        numerical,  # feature 3
        # ... define the features here.
    ]
    return tf.data.experimental.CsvDataset(path, csv_columns, header=True)

  ds_columns = ds_path.interleave(read_csv_file)

  label_values = ["<=50K", ">50K"]

  init_label_table = tf.lookup.KeyValueTensorInitializer(
      keys=tf.constant(label_values),
      values=tf.constant(range(label_values), dtype=tf.int64))

  label_table = tf.lookup.StaticVocabularyTable(
      init_label_table, num_oov_buckets=1)

  def extract_label(*columns):
    return columns[0:-1], label_table.lookup(columns[-1])

  ds_dataset = ds_columns.map(extract_label)
  ds_dataset = ds_dataset.batch(500)
  return ds_dataset


strategy = tf.distribute.experimental.ParameterServerStrategy(...)

with strategy.scope():
  model = tfdf.keras.DistributedGradientBoostedTreesModel()

  train_dataset = strategy.distribute_datasets_from_function(
      lambda context: dataset_fn(context, [...list of csv files...])
  )

model.fit(train_dataset)

print("Trained model")
model.summary()
```

### Using Yggdrasil Decision Forest for both dataset reading and model training

Start a set of GRPC workers on different machines. You can either use:

1.  The YDF worker binary
    ([doc](https://github.com/google/yggdrasil-decision-forests/blob/main/documentation/user_manual.md#grpc-distribute-implementation-recommended))
    available in the
    [YDF release packages](https://github.com/google/yggdrasil-decision-forests/releases).
2.  Use the TF-DF worker binary available in the TF-DF PyPi package.

Both binaries are equivalent and have the same signature. However, unlike YDF
binary, the TF-DF binary requires the TensorFlow .so file.

**Example of how to start the TF-DF worker binary**

```shell
# Locate the installed pypi package of TF-DF.
pip show tensorflow-decision-forests
# Look for the "Location:" path.
LOCATION=...
WORKER_BINARY=${LOCATION}/tensorflow_decision_forests/keras/grpc_worker_main

# Run the worker binary
export LD_LIBRARY_PATH=${LOCATION}/tensorflow && ${WORKER_BINARY} --port=2001
```

**Distributed training**

```python
import tensorflow_decision_forests as tfdf
import tensorflow as tf

deployment_config = tfdf.keras.core.YggdrasilDeploymentConfig()
deployment_config.try_resume_training = True
deployment_config.distribute.implementation_key = "GRPC"
socket_addresses = deployment_config.distribute.Extensions[
    tfdf.keras.core.grpc_pb2.grpc].socket_addresses

# Socket addresses of ":grpc_worker_main" running instances.
socket_addresses.addresses.add(ip="127.0.0.1", port=2001)
socket_addresses.addresses.add(ip="127.0.0.2", port=2001)
socket_addresses.addresses.add(ip="127.0.0.3", port=2001)
socket_addresses.addresses.add(ip="127.0.0.4", port=2001)

model = tfdf.keras.DistributedGradientBoostedTreesModel(
    advanced_arguments=tfdf.keras.AdvancedArguments(
        yggdrasil_deployment_config=deployment_config))

model.fit_on_dataset_path(
    train_path="/path/to/dataset@100000",
    label_key="label_key",
    dataset_format="tfrecord+tfe")

print("Trained model")
model.summary()
```

See Yggdrasil Decision Forests
[supported formats](https://github.com/google/yggdrasil-decision-forests/blob/main/documentation/user_manual.md#dataset-path-and-format)
for the possible values of `dataset_format`.
