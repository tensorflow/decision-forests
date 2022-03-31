# TensorFlow Decision Forests and TensorFlow Serving

This script compiles TF-Serving with TF-Decision Forests. See the
[TensorFlow Decision Forests and TensorFlow Serving guide](https://www.tensorflow.org/decision_forests/tensorflow_serving)
for background information. Results of this script as published as
`tf_serving_linux.zip` on the
[TF-DF GitHub release page](https://github.com/tensorflow/decision-forests/releases).

Compilation

**Note:** Make sure Docker is installed.

```shell
# In a fresh directory
git clone https://github.com/tensorflow/decision-forests.git
git clone https://github.com/tensorflow/serving.git
decision-forests/tools/tf_serving/build_tf_serving_with_tf_df.sh

# Or sudo decision-forests/tools/tf_serving/build_tf_serving_with_tf_df.sh if
# Docker need to be run as sudo.
```

Usage example:

```shell
# Compile or download TF-Serving.
# The result is the `tensorflow_model_server` binary.

# Configure the model path and name.
MODEL_PATH=/tmp/my_saved_model
MODEL_NAME=my_model

# Make sure that MODEL_PATH contains a version serving sub-directory.
# If not, create the "1" directory manually.
# For example, the structure should be:
tree $MODEL_PATH
# /path/to/tf-df/model
# └── 1
#     ├── assets
#     ├── keras_metadata.pb
#     ├── saved_model.pb
#     └── variables

# Start a TF Serving server
tensorflow_model_server \
    --rest_api_port=8501 \
    --model_name=${MODEL_NAME} \
    --model_base_path=${MODEL_PATH}


# Send test requests to the model:
curl http://localhost:8501/v1/models/${MODEL_NAME}:predict -X POST \
    -d '{"instances": [{"age":[39],"workclass":["State-gov"],"fnlwgt":[77516],"education":["Bachelors"],"education_num":[13],"marital_status":["Never-married"],"occupation":["Adm-clerical"],"relationship":["Not-in-family"],"race":["White"],"sex":["Male"],"capital_gain":[2174],"capital_loss":[0],"hours_per_week":[40],"native_country":["United-States"]}]}'
```
