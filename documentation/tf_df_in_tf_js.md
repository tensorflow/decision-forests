# Running TensorFlow Decision Forests models with TensorFlow.js

These instructions explain how to train a TF-DF model and run it on the
web using TensorFlow.js.

## Detailed instructions

### Train a model in TF-DF

To try out this tutorial, you first need a TF-DF model. You can use your own
model or train a model with the
[Beginner's tutorial](https://www.tensorflow.org/decision_forests/tutorials/beginner_colab).

If you simply want to quickly train a model in Google Colab, you can use the
following code snippet.

```python
!pip install tensorflow_decision_forests -U -qq
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd

# Download the dataset, load it into a pandas dataframe and convert it to TensorFlow format.
!wget -q https://storage.googleapis.com/download.tensorflow.org/data/palmer_penguins/penguins.csv -O /tmp/penguins.csv
dataset_df = pd.read_csv("/tmp/penguins.csv")
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(dataset_df, label="species")

# Create and train the model
model_1 = tfdf.keras.GradientBoostedTreesModel()
model_1.fit(train_ds)
```

### Convert the model

The instructions going forward assume that you have saved your TF-DF model under
the path `/tmp/my_saved_model`. Run the following snippet to convert the model
to TensorFlow.js.

```python
!pip install tensorflow tensorflow_decision_forests 'tensorflowjs>=4.4.0'
!pip install tf_keras

# Prepare and load the model with TensorFlow
import tensorflow as tf
import tensorflowjs as tfjs
from google.colab import files

# Save the model in the SavedModel format
tf.saved_model.save(model_1, "/tmp/my_saved_model")

# Convert the SavedModel to TensorFlow.js and save as a zip file
tfjs.converters.tf_saved_model_conversion_v2.convert_tf_saved_model("/tmp/my_saved_model", "./tfjs_model")

# Download the converted TFJS model
!zip -r tfjs_model.zip tfjs_model/
files.download("tfjs_model.zip")
```

When Google Colab finishes running, it downloads the converted TFJS model as a
zip file.. Unzip this file before using it in the next step.

An unzipped Tensorflow.js model consists of a number of files. The example
model contains the following:

- assets.zip
- group1-shard1of1.bin
- model.json


### Use the Tensorflow.js model on the web

Use this template to load TFJS dependencies and run the TFDF model. Change the
model path to where your model is served and modify the tensor given to
executeAsync.

```html
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.5.0/dist/tf.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tfdf/dist/tf-tfdf.min.js"></script>
  <script>
    (async () =>{
      // Load the model.
      // Tensorflow.js currently needs the absolute path to the model including the full origin.
      const model = await tfdf.loadTFDFModel('https://path/to/unzipped/model/model.json');
      // Perform an inference
      const result = await model.executeAsync({
            "island": tf.tensor(["Torgersen"]),
            "bill_length_mm": tf.tensor([39.1]),
            "bill_depth_mm": tf.tensor([17.3]),
            "flipper_length_mm": tf.tensor([3.1]),
            "body_mass_g": tf.tensor([1000.0]),
            "sex": tf.tensor(["Female"]),
            "year": tf.tensor([2007], [1], 'int32'),
      });
      // The result is a 6-dimensional vector, the first half may be ignored
      result.print();
    })();
  </script>
```

## Questions?

Check out the
[TensorFlow Decision Forests documentation](https://www.tensorflow.org/decision_forests)
and the [TensorFlow.js documentation](https://www.tensorflow.org/js/tutorials).
