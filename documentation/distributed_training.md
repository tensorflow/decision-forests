# Distributed Training

**Distributed training** is a type of model training where the computing
resources requirements (e.g., CPU, RAM) are distributed among multiple
computers. Distributed training allows to train faster and on larger datasets
(up to a few billion examples).

Distributed training is also useful for **automated hyper-parameter
optimization** where multiple models are trained in parallel.

In this document you will learn how to:

-   Train a TF-DF model using distributed training.
-   Tune the hyper-parameters of a TF-DF model using distributed training.

## Limitations

As of now, distributed training is supported for:

-   Training Gradient Boosted Trees models with
    `tfdf.keras.DistributedGradientBoostedTreesModel`. Distributed Gradient
    Boosted Trees models are equivalent to their non-distributed counterparts.
-   Hyper-parameter search for any TF-DF model type.

## How to enable distributed training

This section list the steps to enabled distributed training. For full examples,
see the next section.

### ParameterServerStrategy scope

The model and the dataset are defined in a `ParameterServerStrategy` scope.

```python
strategy = tf.distribute.experimental.ParameterServerStrategy(...)
with strategy.scope():
  model = tfdf.keras.DistributedGradientBoostedTreesModel()
  distributed_train_dataset = strategy.distribute_datasets_from_function(dataset_fn)
model.fit(distributed_train_dataset)
```

### Dataset format

Like for non-distributed training, datasets can be provided as

1.  A finite tensorflow distributed dataset, or
2.  a path to the dataset files using one of
    [the compatible dataset formats](https://ydf.readthedocs.io/en/latest/cli_user_manual.html#dataset-path-and-format).

Using sharded files is significantly simpler than using the finite tensorflow
distributed dataset approach (1 line vs ~20 lines of code). However, only the
tensorflow dataset approach supports TensorFlow pre-processing. If your pipeline
does not contain any pre-processing, the sharded dataset option is recommended.

In both cases, the dataset should be sharded into multiple files to distribute
dataset reading efficiently.

### Setup workers

A **chief process** is the program running the python code that defines the
TensorFlow model. This process is not running any heavy computation. The
effective training computation is done by **workers**. Workers are processes
running a TensorFlow Parameter Server.

The chief should be configured with the IP address of the workers. This can be
done using the `TF_CONFIG` environment variable, or by creating a
`ClusterResolver`. See
[Parameter server training with ParameterServerStrategy](https://www.tensorflow.org/tutorials/distribute/parameter_server_training)
for more details.

TensorFlow's ParameterServerStrategy defines two type of workers: "workers" and
"parameter server". TensorFlow requires at least one of each type of worker to
be instantiated. However, TF-DF only uses "workers". So, one "parameter server"
needs to be instantiated but will not be used by TF-DF. For example, the
configuration of a TF-DF training might look as follows:

-   1 Chief
-   50 Workers
-   1 Parameter server

Note: If you use TFX, the configuration of chief/workers/parameter server is
done automatically.

The workers require access to TensorFlow Decision Forests' custom training ops.
There are two options to enable access:

1.  Use the pre-configured TF-DF C++ Parameter Server
    `//third_party/tensorflow_decision_forests/tensorflow/distribute:tensorflow_std_server`.
2.  Create a parameters server by calling `tf.distribute.Server()`. In this
    case, TF-DF should be imported `import tensorflow_decision_forests`.

## Examples

This section shows full examples of distributed training configurations. For
more examples, check the
[TF-DF unit tests](https://github.com/tensorflow/decision-forests/blob/main/tensorflow_decision_forests/keras/keras_distributed_test.py).

### Example: Distributed training on dataset path

Divide your dataset into a set of sharded files using one of
[the compatible dataset formats](https://ydf.readthedocs.io/en/latest/cli_user_manual.html#dataset-path-and-format).
It is recommended to names the files as follows: `/path/to/dataset/train-<5
digit index>-of-<total files>`, for example

```
/path/to/dataset/train-00000-of-00100
/path/to/dataset/train-00001-of-00005
/path/to/dataset/train-00002-of-00005
...
```

For maximum efficiency, the number of files should be at least 10x the number of
workers. For example, if you are training with 100 workers, make sure the
dataset is divided in at least 1000 files.

The files can then be referenced with a sharding expression such as:

-   /path/to/dataset/train@1000
-   /path/to/dataset/train@*

Distributed training is done as follows. In this example, the dataset is stored
as a TFRecord of TensorFlow Examples (defined by the key `tfrecord+tfe`).

```python
import tensorflow_decision_forests as tfdf
import tensorflow as tf

strategy = tf.distribute.experimental.ParameterServerStrategy(...)

with strategy.scope():
  model = tfdf.keras.DistributedGradientBoostedTreesModel()

model.fit_on_dataset_path(
    train_path="/path/to/dataset/train@1000",
    label_key="label_key",
    dataset_format="tfrecord+tfe")

print("Trained model")
model.summary()
```

### Example: Distributed training on a finite TensorFlow distributed dataset

TF-DF expects a distributed finite worker-sharded TensorFlow dataset:

-   **Distributed** : A non-distributed dataset is wrapped in
    `strategy.distribute_datasets_from_function`.
-   **finite**: The dataset should read each example exactly once. The dataset
    should should **not** contain any `repeat` instructions.
-   **worker-sharded**: Each worker should read a separate part of the dataset.

Here is an example:

```python
import tensorflow_decision_forests as tfdf
import tensorflow as tf


def dataset_fn(context, paths):
  """Create a worker-sharded finite dataset from paths.

  Like for non-distributed training, each example should be visited exactly
  once (and by only one worker) during the training. In addition, for optimal
  training speed, the reading of the examples should be distributed among the
  workers (instead of being read by a single worker, or read and discarded
  multiple times).

  In other words, don't add a "repeat" statement and make sure to shard the
  dataset at the file level and not at the example level.
  """

  # List the dataset files
  ds_path = tf.data.Dataset.from_tensor_slices(paths)

  # Make sure the dataset is used with distributed training.
  assert context is not None


  # Split the among the workers.
  #
  # Note: The "shard" is applied on the file path. The shard should not be
  # applied on the examples directly.
  # Note: You cannot use 'context.num_input_pipelines' with ParameterServerV2.
  current_worker = tfdf.keras.get_worker_idx_and_num_workers(context)
  ds_path = ds_path.shard(
      num_shards=current_worker.num_workers,
      index=current_worker.worker_idx)

  def read_csv_file(path):
    """Reads a single csv file."""

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

  # We assume a binary classification label with the following possible values.
  label_values = ["<=50K", ">50K"]

  # Convert the text labels into integers:
  # "<=50K" => 0
  # ">50K" => 1
  init_label_table = tf.lookup.KeyValueTensorInitializer(
      keys=tf.constant(label_values),
      values=tf.constant(range(label_values), dtype=tf.int64))
  label_table = tf.lookup.StaticVocabularyTable(
      init_label_table, num_oov_buckets=1)

  def extract_label(*columns):
    return columns[0:-1], label_table.lookup(columns[-1])

  ds_dataset = ds_columns.map(extract_label)

  # The batch size has no impact on the quality of the model. However, a larger
  # batch size generally is faster.
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

### Example: Distributed hyper-parameter tuning on a dataset path

Distributed hyper-parameter tuning on a dataset path is similar to distributed
training. The only difference is that this option is compatible with
non-distributed models. For example, you can distribute the hyper-parameter
tuning of the (non-distributed) Gradient Boosted Trees model.

```python
with strategy.scope():
  tuner = tfdf.tuner.RandomSearch(num_trials=30, use_predefined_hps=True)
  model = tfdf.keras.GradientBoostedTreesModel(tuner=tuner)

training_history = model.fit_on_dataset_path(
  train_path=train_path,
  label_key=label,
  dataset_format="csv",
  valid_path=test_path)

logging.info("Trained model:")
model.summary()
```

### Example: Unit testing

To unit test distributed training, you can create mock worker processes. See the
method `_create_in_process_tf_ps_cluster` in
[TF-DF unit tests](https://github.com/tensorflow/decision-forests/blob/main/tensorflow_decision_forests/keras/keras_distributed_test.py)
for more information.
