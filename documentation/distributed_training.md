# Distributed Training

<!-- docs_infra:strip_begin -->

## Table of Contents

<!--ts-->

<!--te-->

<!-- docs_infra:strip_end -->

**Warning: Distributed training is experimental in TF-DF.**

## Introduction

Distributed training makes it possible to train models quickly on larger
datasets. Distributed training in TF-DF relies on the TensorFlow
ParameterServerV2 distribution strategy or the Yggdrasil Decision Forest GRPC
distribute strategy. Only some of the TF-DF models support distributed training.

See the
[distributed training](https://github.com/google/yggdrasil-decision-forests/blob/main/documentation/user_manual.md#distributed-training)
section in the Yggdrasil Decision Forests user manual for details about the
available distributed training algorithms. When using distributed training with
TF Parameter Server in TF-DF, Yggdrasil Decision Forests is effectively running
the `TF_DIST` distribute implementation.

**Note:** Currently (Oct. 2021), the shared (i.e. != monolithic) OSS build of
TF-DF does not support TF ParameterServer distribution strategy. Please use the
Yggdrasil DF GRPC distribute strategy instead.

## Dataset

Similarly to the non-distributed training scenario, each example should be
visited exactly once during the training. In addition, for optimal training
speed, the reading of the examples should be distributed among the workers
(instead of being read by a single worker, or read and discarded multiple times)
. The distribution of datasets reading in TF2 is still incomplete.

As of today ( Oct 2021), the following solutions are available for TF-DF:

1.  To use **Yggdrasil Decision Forests distributed dataset reading**. This
    solution is the fastest and the one that gives the best results as it is
    currently the only one that guarantees that each example is read only once.
    The downside is that this solution does not support TensorFlow
    pre-processing. The "Yggdrasil DF GRPC distribute strategy" only support
    this option for dataset reading.

2.  To use **ParameterServerV2 distributed dataset** with dataset file sharding
    using TF-DF worker index. This solution is the most natural for TF users.

Currently, using ParameterServerV2 distributed dataset with context or
tf.data.service are not compatible with TF-DF.

## Examples

Following are some examples of distributed training.

### Distribution with Yggdrasil distributed dataset reading and TF ParameterServerV2 strategy

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

### Distribution with ParameterServerV2 distributed dataset and TF ParameterServerV2 strategy

```python
import tensorflow_decision_forests as tfdf
import tensorflow as tf

global_batch_size = 120
num_train_examples = 123456  # Number of training examples


def dataset_fn(context, paths):
  assert context is not None, "The dataset_fn is not distributed"

  ds_path = tf.data.Dataset.from_tensor_slices(paths)

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
  ds_dataset = ds_dataset.batch(global_batch_size)

  # The "repeat" is currently necessary, but should be removed eventually.
  ds_dataset = ds_dataset.repeat(None)

  return ds_dataset


strategy = tf.distribute.experimental.ParameterServerStrategy(...)

with strategy.scope():
  model = tfdf.keras.DistributedGradientBoostedTreesModel()

  train_dataset = strategy.distribute_datasets_from_function(
      lambda context: dataset_fn(context, [...list of csv files...])
  )

model.fit(
    train_dataset,
    steps_per_epoch=num_train_examples // global_batch_size)

print("Trained model")
model.summary()
```

### Distribution with Yggdrasil distributed dataset reading and Yggdrasil DF GRPC distribute strategy

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
