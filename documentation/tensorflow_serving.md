# TensorFlow Decision Forests and TensorFlow Serving

[TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) (TF Serving)
is a tool to run TensorFlow models online in large production settings using a
RPC or REST API. TensorFlow Decision Forests (TF-DF) is supported natively by TF
Serving >=2.11.

TF-DF models are directly compatible with TF Serving. Yggdrasil models can be
used with TF Serving after being
[converted](https://ydf.readthedocs.io/en/latest/convert_model.html#convert-a-yggdrasil-model-to-a-tensorflow-decision-forests-model)
first.

## Limitations

TensorFlow adds a significant amount of computation overhead. For small, latency
sensitive models (e.g., model inference time ~1µs), this overhead can be
an order of magnitude larger than time needed by the model itself.
In this case, it is recommended to run the TF-DF models with
[Yggdrasil Decision Forests](https://ydf.readthedocs.io).

## Usage example

The following example shows how to run a TF-DF model in TF Serving:

First, [install TF Serving](https://github.com/tensorflow/serving#set-up). In
this example, we will use a pre-compiled version of TF-Serving + TF-DF.

```shell
# Download TF Serving
wget https://github.com/tensorflow/decision-forests/releases/download/serving-1.0.1/tensorflow_model_server_linux.zip
unzip tensorflow_model_server_linux.zip

# Check that TF Serving works.
./tensorflow_model_server --version
```

In this example, we use an already trained TF-DF model.

```shell
# Get a TF-DF model
git clone https://github.com/tensorflow/decision-forests.git
MODEL_PATH=$(pwd)/decision-forests/tensorflow_decision_forests/test_data/model/saved_model_adult_rf

echo "The TF-DF model is available at: ${MODEL_PATH}"
```

**Notes:** TF-Serving requires the model's full path. This is why we use
`$(pwd)`.

TF-Serving supports model versioning. The model should be contained in a
directory whose name is the version of the model. A model version is an integer
e.g., "1". Here is a typical directory for TF-Serving.

-   `/path/to/model`
    -   `1` : Version 1 of the model
    -   `5` : Version 5 of the model
    -   `6` : Version 6 of the model

For this example, we only need to put the model in a directory called "1".

```shell
mkdir -p /tmp/tf_serving_model
cp -R "${MODEL_PATH}" /tmp/tf_serving_model/1
```

Now, we can start TF-Sering on the model.

```shell
./tensorflow_model_server \
    --rest_api_port=8502 \
    --model_name=my_model \
    --model_base_path=/tmp/tf_serving_model
```

Finally, you can send a request to TF Serving using the Rest API. Two formats
are available: predict+instances API and predict+inputs API. Here is an example
of each of them:

```shell
# Predictions with the predict+instances API.
curl http://localhost:8502/v1/models/my_model:predict -X POST \
    -d '{"instances": [{"age":39,"workclass":"State-gov","fnlwgt":77516,"education":"Bachelors","education_num":13,"marital_status":"Never-married","occupation":"Adm-clerical","relationship":"Not-in-family","race":"White","sex":"Male","capital_gain":2174,"capital_loss":0,"hours_per_week":40,"native_country":"United-States"}]}'
```

```shell
# Predictions with the predict+inputs API
curl http://localhost:8502/v1/models/my_model:predict -X POST \
    -d '{"inputs": {"age":[39],"workclass":["State-gov"],"fnlwgt":[77516],"education":["Bachelors"],"education_num":[13],"marital_status":["Never-married"],"occupation":["Adm-clerical"],"relationship":["Not-in-family"],"race":["White"],"sex":["Male"],"capital_gain":[2174],"capital_loss":[0],"hours_per_week":[40],"native_country":["United-States"]}}'
```
