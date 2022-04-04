# TensorFlow Decision Forests and TensorFlow Serving

`build_tf_serving_with_tf_df.sh` is a script that compiles TF-Serving with
TF-Decision Forests. See the
[TensorFlow Decision Forests and TensorFlow Serving guide](https://www.tensorflow.org/decision_forests/tensorflow_serving)
for background information. Results of this script as published as
`tf_serving_linux.zip` on the
[TF-DF GitHub release page](https://github.com/tensorflow/decision-forests/releases).

**Compilation**

*Note:* Make sure Docker is installed.

```shell
# In a fresh directory
git clone https://github.com/tensorflow/decision-forests.git
git clone https://github.com/tensorflow/serving.git
decision-forests/tools/tf_serving/build_tf_serving_with_tf_df.sh

# Or sudo decision-forests/tools/tf_serving/build_tf_serving_with_tf_df.sh if
# Docker need to be run as sudo.
```

**Usage example:**

```shell
# Make sure TF-DF is installed (this is only necessary to train the model)
pip3 install tensorflow_decision_forests -U

# Train a Random Forest model on the adult dataset
# Save the model to "/tmp/my_saved_model"
python3 decision-forests/examples/minimal.py

# Add a version number to the model (this is required for TF Serving)
mkdir -p /tmp/my_saved_model_with_version
cp -r /tmp/my_saved_model /tmp/my_saved_model_with_version/1

# Compile or download TF-Serving, and set the TFSERVING variable accordingly.
# TFServing binary.
TFSERVING="./tensorflow_model_server"

# Configure the model path and name.
MODEL_PATH=/tmp/my_saved_model_with_version
MODEL_NAME=my_model

# Start a TF Serving server
# Note: This command is blocking. You need to run it in a separate terminal (or
# using &).
${TFSERVING} \
    --rest_api_port=8501 \
    --model_name=${MODEL_NAME} \
    --model_base_path=${MODEL_PATH}

# Send requests to the model:
# See https://www.tensorflow.org/tfx/serving/api_rest for the various solutions.

# Predictions with the predict+instances API.
curl http://localhost:8501/v1/models/${MODEL_NAME}:predict -X POST \
    -d '{"instances": [{"age":39,"workclass":"State-gov","fnlwgt":77516,"education":"Bachelors","education_num":13,"marital_status":"Never-married","occupation":"Adm-clerical","relationship":"Not-in-family","race":"White","sex":"Male","capital_gain":2174,"capital_loss":0,"hours_per_week":40,"native_country":"United-States"}]}'

# Predictions with the predict+inputs API
curl http://localhost:8501/v1/models/${MODEL_NAME}:predict -X POST \
    -d '{"inputs": {"age":[39],"workclass":["State-gov"],"fnlwgt":[77516],"education":["Bachelors"],"education_num":[13],"marital_status":["Never-married"],"occupation":["Adm-clerical"],"relationship":["Not-in-family"],"race":["White"],"sex":["Male"],"capital_gain":[2174],"capital_loss":[0],"hours_per_week":[40],"native_country":["United-States"]}}'
```
