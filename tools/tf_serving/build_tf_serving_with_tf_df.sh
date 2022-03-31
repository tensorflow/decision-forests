# Compile TensorFlow Servo with TensorFlow Decision Forests.
#
# This script is equivalent as the instructions at:
# https://www.tensorflow.org/decision_forests/tensorflow_serving
#
# This script is a standalone: It will compile the version TF-DF available
# publicly on github.
#
# Usage example:
#   # In a fresh directory
#   git clone https://github.com/tensorflow/decision-forests.git
#   git clone https://github.com/tensorflow/serving.git
#   decision-forests/tools/tf_serving/build_tf_serving_with_tf_df.sh
#
set -e

# Add TF-DF as a dependency.

# WORKSPACE
WORKSPACE_PATH="serving/WORKSPACE"
if [ ! -f "${WORKSPACE_PATH}.bk" ]; then
  echo "Edit WORKSPACE: ${WORKSPACE_PATH}"
  cp ${WORKSPACE_PATH} ${WORKSPACE_PATH}.bk
  sed -i '3rdecision-forests/tools/tf_serving/build_tf_serving_workspace_extra.txt' ${WORKSPACE_PATH}
else
  echo "WORKSPACE already edited: ${WORKSPACE_PATH}"
fi

# BUILD
BUILD_PATH="serving/tensorflow_serving/model_servers/BUILD"
if [ ! -f "${BUILD_PATH}.bk" ]; then
  echo "Edit BUILD: ${BUILD_PATH}"
  cp ${BUILD_PATH} ${BUILD_PATH}.bk
  sed -i 's|if_v2(\[\])|if_v2([\n    "@org_tensorflow_decision_forests//tensorflow_decision_forests/tensorflow/ops/inference:kernel_and_op"\n])|g' serving/tensorflow_serving/model_servers/BUILD
else
  echo "BUILD already edited: ${BUILD_PATH}"
fi

# Compile Servo
echo "Compile Servo"
(cd serving && tools/run_in_docker.sh bazel build -c opt \
  --copt=-mfma \
  --define use_tensorflow_io=1 \
  --define no_absl_statusor=1 \
  --copt=-mavx2 tensorflow_serving/model_servers:tensorflow_model_server )

# Pack the binary in a zip. This is the zip distributed in github.
BINARY=serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
zip -j tensorflow_model_server_linux.zip ${BINARY} README.md

# If you have a model, you can run it in TF Serving using the following command:
# ${BINARY} \
# --rest_api_port=8501 \
# --model_name=my_model \
# --model_base_path=/path/to/tfdf/model
