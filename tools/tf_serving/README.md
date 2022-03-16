# TensorFlow Decision Forests and TensorFlow Serving

Guide: https://www.tensorflow.org/decision_forests/tensorflow_serving

Generator:
https://github.com/google/yggdrasil-decision-forests/tree/main/tools/tf_serving

Usage example:

```
# Start a TF Serving server

tensorflow_model_server \
    --rest_api_port=8501 \
    --model_name=my_model \
    --model_base_path=/path/to/tfdf/model
```
