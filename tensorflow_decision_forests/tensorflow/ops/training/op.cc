/*
 * Copyright 2021 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// TensorFlow Ops for the training of Yggdrasil models.

#include "tensorflow/core/framework/op.h"

#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

// Loads an Yggdrasil model in memory from a disk path.
//
// The model is loaded in a tensorflow resource managed by the TensorFlow's
// ResourceMgr. The model is not compiled.
//
// OP: FileModelLoader
//
// Args:
//   model_path: (tensor(string)) Path to an Yggdrasil model.
//   model_identifier: (string) Unique resource identifier of the model.
//
// Returns:
//   Nothing.
//
// Dependencies:
//
// You binary should also include the Yggdrasil model library corresponding
// to your model. For example,
// "//third_party/yggdrasil_decision_forests/model/random_forest"
// in the case of a Random Forest model.
//
REGISTER_OP("SimpleMLFileModelLoader")
    .SetIsStateful()
    .Input("model_path: string")
    .Attr("model_identifier: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      return OkStatus();
    });

// Feature value collector.
//
// Collects a feature values and stores them in a shared TF resource with the
// given "id" name.

// Numerical value.
REGISTER_OP("SimpleMLNumericalFeature")
    .SetIsStateful()
    .Input("value: float32")
    .Attr("id: string")
    .Attr("feature_name : string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      return OkStatus();
    });

// Categorical value stored as a string.
REGISTER_OP("SimpleMLCategoricalStringFeature")
    .SetIsStateful()
    .Input("value: string")
    .Attr("id: string")
    .Attr("feature_name : string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      return OkStatus();
    });

// Categorical value stored as a positive integer.
REGISTER_OP("SimpleMLCategoricalIntFeature")
    .SetIsStateful()
    .Input("value: int32")
    .Attr("id: string")
    .Attr("feature_name : string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      return OkStatus();
    });

// Categorical-set value stored as a string.
REGISTER_OP("SimpleMLCategoricalSetStringFeature")
    .SetIsStateful()
    .Input("values: string")
    .Input("row_splits: int64")
    .Attr("id: string")
    .Attr("feature_name : string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      return OkStatus();
    });

// Categorical-set value stored as a positive integer.
REGISTER_OP("SimpleMLCategoricalSetIntFeature")
    .SetIsStateful()
    .Input("values: int32")
    .Input("row_splits: int64")
    .Attr("id: string")
    .Attr("feature_name : string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      return OkStatus();
    });

// Hash value.
// The string value is hashed using "dataset::HashColumnString".
REGISTER_OP("SimpleMLHashFeature")
    .SetIsStateful()
    .Input("value: string")
    .Attr("id: string")
    .Attr("feature_name : string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      return OkStatus();
    });

// Feature value collector on file.
//
// Collects feature values export them to a file. Each worker stores values
// in one file, but the whole set of values for a feature will be stored in
// various files (one per worker in general).

// Numerical value.
REGISTER_OP("SimpleMLNumericalFeatureOnFile")
    .SetIsStateful()
    .Input("value: float32")
    .Attr("resource_id: string")
    .Attr("feature_idx: int")
    .Attr("feature_name : string")
    .Attr("dataset_path : string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      return OkStatus();
    });

// Categorical int value.
REGISTER_OP("SimpleMLCategoricalIntFeatureOnFile")
    .SetIsStateful()
    .Input("value: int32")
    .Attr("resource_id: string")
    .Attr("feature_idx: int")
    .Attr("feature_name : string")
    .Attr("dataset_path : string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      return OkStatus();
    });

// Categorical string value.
REGISTER_OP("SimpleMLCategoricalStringFeatureOnFile")
    .SetIsStateful()
    .Input("value: string")
    .Attr("resource_id: string")
    .Attr("feature_idx: int")
    .Attr("feature_name : string")
    .Attr("dataset_path : string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      return OkStatus();
    });

// Finalizes the reading of values and storage on the file-system on a worker.
// Should be called on each worker (i.e. any instance having a ...FeatureOnFile
// op) and before `SimpleMLChiefFinalizeDiskFeature`
REGISTER_OP("SimpleMLWorkerFinalizeFeatureOnFile")
    .Attr("feature_resource_ids: list(string)")
    .Attr("dataset_path: string")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      return OkStatus();
    });

// Finalizes the acquisition of values from the disk features. Should be called
// after `SimpleMLWorkerFinalizeDiskFeature`.
REGISTER_OP("SimpleMLChiefFinalizeFeatureOnFile")
    .Attr("feature_names: list(string)")
    .Attr("dataset_path: string")
    .Attr("num_shards : int")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      return OkStatus();
    });

// Trains a model a dataset stored in RAM i.e. feature values collected by
// "SimpleML*Feature" OPs.
//
// This operation has no output tensors. Instead, the output model is stored in
// the resource, and exported in the "model_dir" (if specified).
//
// Args:
//   feature_ids: Comma separated list of feature accumulator ids.
//   label_id: Feature accumulator containing the labels.
//   weight_id: Feature accumulator containing the example weight. Empty string
//     if no example weight are available.
//   model_id: If set, exports the trained model in a resource called
//     "model_id".
//   model_dir: Export directory for the training logs and model. If not
//     provided, the logs/model are/is not exported.
//   learner: Yggdrasil learner name.
//   hparams: Serialized abstract_learner_pb2.GenericHyperParameters proto.
//   Hyper-parameters of the model.
//   task: Task solved by the model. Integer casting of a proto::Task.
//   training_config: Serialized TrainingConfig proto (See learning/lib/ami/
//     simple_ml/learner/abstract_learner.proto). Specifies the hparams of
//     the model similarly to hparams. "hparams" has priority over
//     "training_config". Note: "training_config" has greater coverage than
//     "hparams". The fields "label", "weight", "task", "features" and
//     "learner" should be left empty. Set "label" and "task" using the Op
//     parameters.
//   deployment_config: Serialized DeploymentConfig proto (See learning/lib/ami/
//     simple_ml/learner/abstract_learner.proto). Specifies computation
//     resources to use for the training. By default, the training is done
//     locally using 6 threads.
//   guide: Serialized DataSpecificationGuide proto (See
//     //third_party/yggdrasil_decision_forests/dataset/data_spec.proto).
//   has_validation_dataset: If true, a validation dataset is available. The
//     name of the tf resources containing the data are similar to the one of
//     the training dataset with the "__VALIDATION" postfix.
//   use_file_prefix: If true, the model files are prefixed with the model_id.
//     For internal use only.
//
// Output:
//   success: True iif. the training succeeded. The op can fail if: There are
//     not training examples.
REGISTER_OP("SimpleMLModelTrainer")
    .SetIsStateful()
    .Attr("feature_ids: string")
    .Attr("label_id: string")
    .Attr("weight_id: string")
    .Attr("model_id: string")
    .Attr("model_dir: string")
    .Attr("learner: string")
    .Attr("hparams: string")
    .Attr("task: int")
    .Attr("training_config: string")
    .Attr("deployment_config: string")
    .Attr("guide: string = ''")
    .Attr("has_validation_dataset: bool = false")
    .Attr("use_file_prefix: bool = false")
    .Output("success: bool")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return OkStatus();
    });

// Trains a model on a dataset stored on file.
//
// Support any dataset format supported by Yggdrasil Decision Forests. Possibly,
// can train on the "partial dataset cache" format previously generated by
// "SimpleML*FeatureOnFile" OPs.
//
// This operation has no output tensors. Instead, the output model is stored in
// the resource, and exported in the "model_dir" (if specified).
//
REGISTER_OP("SimpleMLModelTrainerOnFile")
    .SetIsStateful()
    .Attr("train_dataset_path: string")
    .Attr("valid_dataset_path: string = ''")
    .Attr("model_id: string")
    .Attr("model_dir: string")
    .Attr("hparams: string")
    .Attr("training_config: string")
    .Attr("deployment_config: string")
    .Attr("guide: string = ''")
    .Attr("use_file_prefix: bool = false")
    .Output("success: bool")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return OkStatus();
    });

// Built a human readable description of the model.
//
// Args:
//   model_identifier: Resource id of the model.
//
// Output:
//   output: Human readable description of the model.
REGISTER_OP("SimpleMLShowModel")
    .SetIsStateful()
    .Attr("model_identifier: string")
    .Output("description: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return OkStatus();
    });

// Unload a model from memory.
//
// Args:
//   model_identifier: Resource id of the model.
//
// Output:
//   output: Serialized AbstractModel. Empty string if the model is missing.
REGISTER_OP("SimpleMLUnloadModel")
    .SetIsStateful()
    .Attr("model_identifier: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      return OkStatus();
    });

// Sets the level of logging in the Yggdrasil decision forest code i.e. calls
// yggdrasil_decision_forests::logging::SetLoggingLevel.
// This is a no-op for the internal build.
//
// Args:
//   level: Level of logging.
//
// Returns:
//   Nothing.
//
REGISTER_OP("YggdrasilDecisionForestsSetLoggingLevel")
    .SetIsStateful()
    .Attr("level: int")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      return OkStatus();
    });

}  // namespace tensorflow
