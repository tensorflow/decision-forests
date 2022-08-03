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

// Trains a model from a dataset stored on the file-system.
//
// Support any of the Yggdrasil Decision Forests dataset formats (as long as the
// dataset reader is registered).
//

#include <algorithm>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow_decision_forests/tensorflow/ops/training/feature_on_file.h"
#include "tensorflow_decision_forests/tensorflow/ops/training/kernel.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/tensorflow.h"

namespace tensorflow_decision_forests {
namespace ops {

namespace tf = ::tensorflow;
namespace model = ::yggdrasil_decision_forests::model;
namespace utils = ::yggdrasil_decision_forests::utils;
namespace dataset = ::yggdrasil_decision_forests::dataset;

REGISTER_KERNEL_BUILDER(
    Name("SimpleMLNumericalFeatureOnFile").Device(tf::DEVICE_CPU),
    SimpleMLNumericalFeatureOnFile);

REGISTER_KERNEL_BUILDER(
    Name("SimpleMLCategoricalIntFeatureOnFile").Device(tf::DEVICE_CPU),
    SimpleMLCategoricalIntFeatureOnFile);

REGISTER_KERNEL_BUILDER(
    Name("SimpleMLCategoricalStringFeatureOnFile").Device(tf::DEVICE_CPU),
    SimpleMLCategoricalStringFeatureOnFile);

REGISTER_KERNEL_BUILDER(
    Name("SimpleMLWorkerFinalizeFeatureOnFile").Device(tf::DEVICE_CPU),
    SimpleMLWorkerFinalizeFeatureOnFile);

REGISTER_KERNEL_BUILDER(
    Name("SimpleMLChiefFinalizeFeatureOnFile").Device(tf::DEVICE_CPU),
    SimpleMLChiefFinalizeFeatureOnFile);

// Trains a simpleML model.
class SimpleMLModelTrainerOnFile : public tensorflow::OpKernel {
 public:
  explicit SimpleMLModelTrainerOnFile(tf::OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("train_dataset_path", &train_dataset_path_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("valid_dataset_path", &valid_dataset_path_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("model_dir", &model_dir_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("model_id", &model_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_file_prefix", &use_file_prefix_));

    std::string serialized_guide;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("guide", &serialized_guide));
    if (!guide_.ParseFromString(serialized_guide)) {
      OP_REQUIRES_OK(ctx, tf::Status(tf::error::INVALID_ARGUMENT,
                                     "Cannot de-serialize guide proto."));
    }

    std::string hparams;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("hparams", &hparams));
    if (!hparams_.ParseFromString(hparams)) {
      OP_REQUIRES_OK(ctx, tf::Status(tf::error::INVALID_ARGUMENT,
                                     "Cannot de-serialize hparams proto."));
    }

    {
      std::string serialized_training_config;
      OP_REQUIRES_OK(
          ctx, ctx->GetAttr("training_config", &serialized_training_config));
      if (!training_config_.MergeFromString(serialized_training_config)) {
        OP_REQUIRES_OK(
            ctx, tf::Status(tf::error::INVALID_ARGUMENT,
                            "Cannot de-serialize training_config proto."));
      }
    }

    {
      std::string serialized_deployment_config;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("deployment_config",
                                       &serialized_deployment_config));
      if (!deployment_config_.MergeFromString(serialized_deployment_config)) {
        OP_REQUIRES_OK(
            ctx, tf::Status(tf::error::INVALID_ARGUMENT,
                            "Cannot de-serialize deployment_config proto."));
      }
    }
  }

  ~SimpleMLModelTrainerOnFile() override = default;

  void Compute(tf::OpKernelContext* ctx) override {
    LOG(INFO) << "Start Yggdrasil model training from disk";

    tf::Tensor* success_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, tf::TensorShape({}), &success_tensor));
    auto success = success_tensor->scalar<bool>();
    success() = true;

    // TODO: Cache the dataspec.
    dataset::proto::DataSpecification data_spec;
    dataset::CreateDataSpec(train_dataset_path_, false, guide_, &data_spec);
    LOG(INFO) << "Dataset:\n" << dataset::PrintHumanReadable(data_spec, false);

    std::unique_ptr<model::AbstractLearner> learner;
    OP_REQUIRES_OK(
        ctx, utils::FromUtilStatus(GetLearner(training_config_, &learner)));
    OP_REQUIRES_OK(
        ctx, utils::FromUtilStatus(learner->SetHyperParameters(hparams_)));
    *learner->mutable_deployment() = deployment_config_;
    if (!model_dir_.empty()) {
      learner->set_log_directory(tf::io::JoinPath(model_dir_, "train_logs"));
    }

    LOG(INFO) << "Training config:\n"
              << learner->training_config().DebugString();

    LOG(INFO) << "Deployment config:\n" << learner->deployment().DebugString();

#ifdef TFDF_STOP_TRAINING_ON_INTERRUPT
    OP_REQUIRES_OK(ctx, interruption::EnableUserInterruption());
    learner->set_stop_training_trigger(&interruption::stop_training);
#endif

    LOG(INFO) << "Train model";
    absl::optional<std::string> valid_dataset_path;
    if (!valid_dataset_path_.empty()) {
      valid_dataset_path = valid_dataset_path_;
    }
    auto model = learner->TrainWithStatus(train_dataset_path_, data_spec,
                                          valid_dataset_path);

#ifdef TFDF_STOP_TRAINING_ON_INTERRUPT
    OP_REQUIRES_OK(ctx, interruption::DisableUserInterruption());
#endif

    OP_REQUIRES_OK(ctx, utils::FromUtilStatus(model.status()));

    // Export model to disk.
    if (!model_dir_.empty()) {
      if (use_file_prefix_) {
        LOG(INFO) << "Export model in log directory: " << model_dir_
                  << " with prefix " << model_id_;
        OP_REQUIRES_OK(
            ctx, utils::FromUtilStatus(SaveModel(
                     tf::io::JoinPath(model_dir_, "model"), model.value().get(),
                     {/*.file_prefix =*/model_id_})));
      } else {
        LOG(INFO) << "Export model in log directory: " << model_dir_
                  << " without prefix";
        OP_REQUIRES_OK(ctx, utils::FromUtilStatus(
                                SaveModel(tf::io::JoinPath(model_dir_, "model"),
                                          model.value().get())));
      }
    }

    // Export model to model resource.
    if (!model_id_.empty()) {
      LOG(INFO) << "Save model in resources";
      auto* model_container = new YggdrasilModelContainer();
      *model_container->mutable_model() = std::move(model.value());
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Create(
                              kModelContainer, model_id_, model_container));
    }
  }

 private:
  std::string model_dir_;
  std::string model_id_;
  std::string train_dataset_path_;
  std::string valid_dataset_path_;
  bool use_file_prefix_;

  model::proto::GenericHyperParameters hparams_;
  model::proto::TrainingConfig training_config_;
  model::proto::DeploymentConfig deployment_config_;
  dataset::proto::DataSpecificationGuide guide_;
};

REGISTER_KERNEL_BUILDER(
    Name("SimpleMLModelTrainerOnFile").Device(tf::DEVICE_CPU),
    SimpleMLModelTrainerOnFile);

}  // namespace ops
}  // namespace tensorflow_decision_forests
