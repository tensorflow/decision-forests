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

// Training ops.

#include "tensorflow_decision_forests/tensorflow/ops/training/kernel.h"

#include <algorithm>

#include "absl/strings/substitute.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow_decision_forests/tensorflow/ops/training/features.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/tensorflow.h"

namespace tensorflow_decision_forests {
namespace ops {

namespace tf = ::tensorflow;
namespace model = ::yggdrasil_decision_forests::model;
namespace utils = ::yggdrasil_decision_forests::utils;
namespace dataset = ::yggdrasil_decision_forests::dataset;

tensorflow::Status YggdrasilModelContainer::LoadModel(
    const absl::string_view model_path) {
  TF_RETURN_IF_ERROR_FROM_ABSL_STATUS(model::LoadModel(model_path, &model_));
  // Cache label information.
  const auto& label_spec = model_->data_spec().columns(model_->label_col_idx());
  num_label_classes_ = label_spec.categorical().number_of_unique_values();
  output_class_representation_.reserve(num_label_classes_);
  for (int class_idx = 0; class_idx < num_label_classes_; class_idx++) {
    output_class_representation_.push_back(
        yggdrasil_decision_forests::dataset::CategoricalIdxToRepresentation(
            label_spec, class_idx));
  }

  LOG(INFO) << "Loading model from " << model_path;
  return tf::Status::OK();
}

// OP loading a model from disk to memory.
class SimpleMLFileModelLoader : public tf::OpKernel {
 public:
  explicit SimpleMLFileModelLoader(tf::OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("model_identifier", &model_identifier_));
  }

  void Compute(tf::OpKernelContext* ctx) override {
    const tf::Tensor& model_path_tensor = ctx->input(0);
    const auto model_paths = model_path_tensor.flat<tf::tstring>();
    CHECK_EQ(model_paths.size(), 1);
    const std::string model_path = model_paths(0);

    auto* model_container = new YggdrasilModelContainer();
    const auto load_model_status = model_container->LoadModel(model_path);
    if (!load_model_status.ok()) {
      model_container->Unref();
      OP_REQUIRES_OK(ctx, load_model_status);
    }

    OP_REQUIRES_OK(
        ctx, ctx->resource_manager()->Create(kModelContainer, model_identifier_,
                                             model_container));
  }

 private:
  std::string model_identifier_;
};

REGISTER_KERNEL_BUILDER(Name("SimpleMLFileModelLoader").Device(tf::DEVICE_CPU),
                        SimpleMLFileModelLoader);

REGISTER_KERNEL_BUILDER(Name("SimpleMLNumericalFeature").Device(tf::DEVICE_CPU),
                        SimpleMLNumericalFeature);

REGISTER_KERNEL_BUILDER(
    Name("SimpleMLCategoricalStringFeature").Device(tf::DEVICE_CPU),
    SimpleMLCategoricalStringFeature);

REGISTER_KERNEL_BUILDER(
    Name("SimpleMLCategoricalIntFeature").Device(tf::DEVICE_CPU),
    SimpleMLCategoricalIntFeature);

REGISTER_KERNEL_BUILDER(
    Name("SimpleMLCategoricalSetStringFeature").Device(tf::DEVICE_CPU),
    SimpleMLCategoricalSetStringFeature);

REGISTER_KERNEL_BUILDER(
    Name("SimpleMLCategoricalSetIntFeature").Device(tf::DEVICE_CPU),
    SimpleMLCategoricalSetIntFeature);

REGISTER_KERNEL_BUILDER(Name("SimpleMLHashFeature").Device(tf::DEVICE_CPU),
                        SimpleMLHashFeature);

FeatureSet::~FeatureSet() { Unlink().IgnoreError(); }

const std::string& FeatureSet::label_feature() const { return label_feature_; }

const std::string& FeatureSet::weight_feature() const {
  return weight_feature_;
}

const std::vector<std::string>& FeatureSet::input_features() const {
  return input_features_;
}

tf::Status FeatureSet::Link(
    tf::OpKernelContext* ctx, const std::string& concat_feature_ids,
    const std::string& label_id, const std::string& weight_id,
    const dataset::proto::DataSpecification* const existing_dataspec) {
  std::vector<std::string> feature_ids =
      absl::StrSplit(concat_feature_ids, ',');
  std::sort(feature_ids.begin(), feature_ids.end());

  if (!label_id.empty()) {
    feature_ids.push_back(label_id);
  }

  if (!weight_id.empty()) {
    feature_ids.push_back(weight_id);
  }

  for (const auto& feature_id : feature_ids) {
    AbstractFeatureResource* feature;
    TF_RETURN_IF_ERROR(
        ctx->resource_manager()->Lookup<AbstractFeatureResource, true>(
            kModelContainer, feature_id, &feature));

    const int feature_idx =
        existing_dataspec ? dataset::GetColumnIdxFromName(
                                feature->feature_name(), *existing_dataspec)
                          : NumFeatures();

    if (feature_id == label_id) {
      label_feature_idx_ = feature_idx;
      label_feature_ = feature->feature_name();
    } else if (feature_id == weight_id) {
      weight_feature_ = feature->feature_name();
    } else {
      input_features_.push_back(feature->feature_name());
    }

    auto* numerical_feature =
        dynamic_cast<SimpleMLNumericalFeature::Resource*>(feature);
    auto* categorical_string_feature =
        dynamic_cast<SimpleMLCategoricalStringFeature::Resource*>(feature);
    auto* categorical_int_feature =
        dynamic_cast<SimpleMLCategoricalIntFeature::Resource*>(feature);
    auto* hash_feature = dynamic_cast<SimpleMLHashFeature::Resource*>(feature);
    auto* categorical_set_string_feature =
        dynamic_cast<SimpleMLCategoricalSetStringFeature::Resource*>(feature);
    auto* categorical_set_int_feature =
        dynamic_cast<SimpleMLCategoricalSetIntFeature::Resource*>(feature);

    if (numerical_feature) {
      numerical_features_.push_back({feature_idx, numerical_feature});
    } else if (categorical_string_feature) {
      categorical_string_features_.push_back(
          {feature_idx, categorical_string_feature});
    } else if (categorical_int_feature) {
      categorical_int_features_.push_back(
          {feature_idx, categorical_int_feature});
    } else if (categorical_set_string_feature) {
      categorical_set_string_features_.push_back(
          {feature_idx, categorical_set_string_feature});
    } else if (categorical_set_int_feature) {
      categorical_set_int_features_.push_back(
          {feature_idx, categorical_set_int_feature});
    } else if (hash_feature) {
      hash_features_.push_back({feature_idx, hash_feature});
    } else {
      return tf::Status(tf::error::Code::INVALID_ARGUMENT,
                        absl::StrCat("Unsupported type for feature \"",
                                     feature->feature_name(), "\""));
    }
  }

  if (!weight_id.empty() && weight_feature_.empty()) {
    return tf::Status(tf::error::Code::INVALID_ARGUMENT,
                      absl::StrCat("Weight feature not found: ", weight_id));
  }

  return tf::Status::OK();
}

tf::Status FeatureSet::IterateFeatures(
    FeatureIterator<SimpleMLNumericalFeature> lambda_numerical,
    FeatureIterator<SimpleMLCategoricalStringFeature> lambda_categorical_string,
    FeatureIterator<SimpleMLCategoricalIntFeature> lambda_categorical_int,
    FeatureIterator<SimpleMLCategoricalSetStringFeature>
        lambda_categorical_set_string,
    FeatureIterator<SimpleMLCategoricalSetIntFeature>
        lambda_categorical_set_int,
    FeatureIterator<SimpleMLHashFeature> lambda_hash) {
  for (auto& feature : numerical_features_) {
    tf::mutex_lock l(*feature.second->mutable_mutex());
    TF_RETURN_IF_ERROR(lambda_numerical(feature.second, feature.first));
  }
  for (auto& feature : categorical_string_features_) {
    tf::mutex_lock l(*feature.second->mutable_mutex());
    TF_RETURN_IF_ERROR(
        lambda_categorical_string(feature.second, feature.first));
  }
  for (auto& feature : categorical_int_features_) {
    tf::mutex_lock l(*feature.second->mutable_mutex());
    TF_RETURN_IF_ERROR(lambda_categorical_int(feature.second, feature.first));
  }
  for (auto& feature : categorical_set_string_features_) {
    tf::mutex_lock l(*feature.second->mutable_mutex());
    TF_RETURN_IF_ERROR(
        lambda_categorical_set_string(feature.second, feature.first));
  }
  for (auto& feature : categorical_set_int_features_) {
    tf::mutex_lock l(*feature.second->mutable_mutex());
    TF_RETURN_IF_ERROR(
        lambda_categorical_set_int(feature.second, feature.first));
  }
  for (auto& feature : hash_features_) {
    tf::mutex_lock l(*feature.second->mutable_mutex());
    TF_RETURN_IF_ERROR(lambda_hash(feature.second, feature.first));
  }

  return tf::Status::OK();
}

tf::Status FeatureSet::Unlink() {
  TF_RETURN_IF_ERROR(IterateFeatures(
      [](SimpleMLNumericalFeature::Resource* feature, const int feature_idx) {
        feature->Unref();
        return tf::Status::OK();
      },
      [](SimpleMLCategoricalStringFeature::Resource* feature,
         const int feature_idx) {
        feature->Unref();
        return tf::Status::OK();
      },
      [](SimpleMLCategoricalIntFeature::Resource* feature,
         const int feature_idx) {
        feature->Unref();
        return tf::Status::OK();
      },
      [](SimpleMLCategoricalSetStringFeature::Resource* feature,
         const int feature_idx) {
        feature->Unref();
        return tf::Status::OK();
      },
      [](SimpleMLCategoricalSetIntFeature::Resource* feature,
         const int feature_idx) {
        feature->Unref();
        return tf::Status::OK();
      },
      [](SimpleMLHashFeature::Resource* feature, const int feature_idx) {
        feature->Unref();
        return tf::Status::OK();
      }));
  numerical_features_.clear();
  categorical_string_features_.clear();
  categorical_int_features_.clear();
  categorical_set_string_features_.clear();
  categorical_set_int_features_.clear();
  hash_features_.clear();
  return tf::Status::OK();
}

// Initialize a dataset (including the dataset's dataspec) from the linked
// resource aggregators.
tf::Status FeatureSet::InitializeDatasetFromFeatures(
    tf::OpKernelContext* ctx,
    const dataset::proto::DataSpecificationGuide& guide,
    dataset::VerticalDataset* dataset) {
  int64_t num_batches = -1;
  int64_t num_examples = -1;
  const auto set_num_examples =
      [&num_examples, &num_batches](
          const int64_t observed_num_examples,
          const int64_t observed_num_batches) -> tf::Status {
    if (num_examples == -1) {
      num_examples = observed_num_examples;
      num_batches = observed_num_batches;
      return tf::Status::OK();
    }
    if (num_examples != observed_num_examples) {
      return tf::Status(
          tf::error::Code::INVALID_ARGUMENT,
          absl::Substitute("Inconsistent number of training examples for the "
                           "different input features $0 != $1.",
                           num_examples, observed_num_examples));
    }
    return tf::Status::OK();
  };

  for (int feature_idx = 0; feature_idx < NumFeatures(); feature_idx++) {
    dataset->mutable_data_spec()->add_columns();
  }

  // Apply the guide on a column. The type of the column should be set.
  const auto apply_guide = [&](const absl::string_view feature_name,
                               dataset::proto::Column* col) -> tf::Status {
    dataset::proto::ColumnGuide col_guide;
    dataset::BuildColumnGuide(feature_name, guide, &col_guide);
    return utils::FromUtilStatus(
        dataset::UpdateSingleColSpecWithGuideInfo(col_guide, col));
  };

  TF_RETURN_IF_ERROR(IterateFeatures(
      [&](SimpleMLNumericalFeature::Resource* feature, const int feature_idx) {
        auto* col = dataset->mutable_data_spec()->mutable_columns(feature_idx);
        col->set_name(feature->feature_name());
        col->set_type(dataset::proto::ColumnType::NUMERICAL);
        TF_RETURN_IF_ERROR(apply_guide(feature->feature_name(), col));
        return set_num_examples(feature->data().size(), feature->NumBatches());
      },
      [&](SimpleMLCategoricalStringFeature::Resource* feature,
          const int feature_idx) {
        auto* col = dataset->mutable_data_spec()->mutable_columns(feature_idx);
        col->set_name(feature->feature_name());
        col->set_type(dataset::proto::ColumnType::CATEGORICAL);
        TF_RETURN_IF_ERROR(apply_guide(feature->feature_name(), col));
        TF_RETURN_IF_ERROR(set_num_examples(feature->indexed_data().size(),
                                            feature->NumBatches()));

        // Don't prune the label feature vocabulary.
        if (feature->feature_name() == label_feature_) {
          col->mutable_categorical()->set_min_value_count(1);
          col->mutable_categorical()->set_max_number_of_unique_values(-1);
        }

        return tf::Status::OK();
      },
      [&](SimpleMLCategoricalIntFeature::Resource* feature,
          const int feature_idx) {
        auto* col = dataset->mutable_data_spec()->mutable_columns(feature_idx);
        col->set_name(feature->feature_name());
        col->set_type(dataset::proto::ColumnType::CATEGORICAL);
        TF_RETURN_IF_ERROR(apply_guide(feature->feature_name(), col));
        col->mutable_categorical()->set_is_already_integerized(true);
        return set_num_examples(feature->data().size(), feature->NumBatches());
      },
      [&](SimpleMLCategoricalSetStringFeature::Resource* feature,
          const int feature_idx) {
        auto* col = dataset->mutable_data_spec()->mutable_columns(feature_idx);
        col->set_name(feature->feature_name());
        col->set_type(dataset::proto::ColumnType::CATEGORICAL_SET);
        TF_RETURN_IF_ERROR(apply_guide(feature->feature_name(), col));
        return set_num_examples(feature->num_examples(),
                                feature->num_batches());
      },
      [&](SimpleMLCategoricalSetIntFeature::Resource* feature,
          const int feature_idx) {
        auto* col = dataset->mutable_data_spec()->mutable_columns(feature_idx);
        col->set_name(feature->feature_name());
        col->set_type(dataset::proto::ColumnType::CATEGORICAL_SET);
        TF_RETURN_IF_ERROR(apply_guide(feature->feature_name(), col));
        col->mutable_categorical()->set_is_already_integerized(true);
        return set_num_examples(feature->num_examples(),
                                feature->num_batches());
      },
      [&](SimpleMLHashFeature::Resource* feature, const int feature_idx) {
        auto* col = dataset->mutable_data_spec()->mutable_columns(feature_idx);
        col->set_name(feature->feature_name());
        col->set_type(dataset::proto::ColumnType::HASH);
        TF_RETURN_IF_ERROR(apply_guide(feature->feature_name(), col));
        return set_num_examples(feature->data().size(), feature->NumBatches());
      }));

  LOG(INFO) << "Number of batches: " << num_batches;
  LOG(INFO) << "Number of examples: " << num_examples;

  if (num_examples <= 0) {
    return tf::Status(tf::error::Code::INVALID_ARGUMENT,
                      "No training examples available.");
  }

  TF_RETURN_IF_ERROR_FROM_ABSL_STATUS(dataset->CreateColumnsFromDataspec());

  dataset->mutable_data_spec()->set_created_num_rows(num_examples);

  dataset::proto::DataSpecificationAccumulator accumulator;
  dataset::InitializeDataspecAccumulator(dataset->data_spec(), &accumulator);

  TF_RETURN_IF_ERROR(IterateFeatures(
      [&](SimpleMLNumericalFeature::Resource* feature, const int feature_idx) {
        auto* col = dataset->mutable_data_spec()->mutable_columns(feature_idx);
        auto* col_acc = accumulator.mutable_columns(feature_idx);
        for (const auto value : feature->data()) {
          TF_RETURN_IF_ERROR_FROM_ABSL_STATUS(
              dataset::UpdateNumericalColumnSpec(value, col, col_acc));
        }
        return tf::Status::OK();
      },
      [&](SimpleMLCategoricalStringFeature::Resource* feature,
          const int feature_idx) {
        auto* col = dataset->mutable_data_spec()->mutable_columns(feature_idx);
        auto* col_acc = accumulator.mutable_columns(feature_idx);
        const auto& reverse_index = feature->reverse_index();
        for (const auto indexed_value : feature->indexed_data()) {
          TF_RETURN_IF_ERROR_FROM_ABSL_STATUS(
              dataset::UpdateCategoricalStringColumnSpec(
                  reverse_index[indexed_value], col, col_acc));
        }
        return tf::Status::OK();
      },
      [&](SimpleMLCategoricalIntFeature::Resource* feature,
          const int feature_idx) {
        auto* col = dataset->mutable_data_spec()->mutable_columns(feature_idx);
        auto* col_acc = accumulator.mutable_columns(feature_idx);
        for (const auto value : feature->data()) {
          TF_RETURN_IF_ERROR_FROM_ABSL_STATUS(
              dataset::UpdateCategoricalIntColumnSpec(value, col, col_acc));
        }
        return tf::Status::OK();
      },
      [&](SimpleMLCategoricalSetStringFeature::Resource* feature,
          const int feature_idx) {
        auto* col = dataset->mutable_data_spec()->mutable_columns(feature_idx);
        auto* col_acc = accumulator.mutable_columns(feature_idx);
        for (const auto& value : feature->values()) {
          TF_RETURN_IF_ERROR_FROM_ABSL_STATUS(
              dataset::UpdateCategoricalStringColumnSpec(value, col, col_acc));
        }
        return tf::Status::OK();
      },
      [&](SimpleMLCategoricalSetIntFeature::Resource* feature,
          const int feature_idx) {
        auto* col = dataset->mutable_data_spec()->mutable_columns(feature_idx);
        auto* col_acc = accumulator.mutable_columns(feature_idx);
        for (const auto value : feature->values()) {
          TF_RETURN_IF_ERROR_FROM_ABSL_STATUS(
              dataset::UpdateCategoricalIntColumnSpec(value, col, col_acc));
        }
        return tf::Status::OK();
      },
      [&](SimpleMLHashFeature::Resource* feature, const int feature_idx) {
        // Nothing to do.
        return tf::Status::OK();
      }));

  dataset::FinalizeComputeSpec({}, accumulator, dataset->mutable_data_spec());

  return tf::Status::OK();
}

tf::Status FeatureSet::MoveExamplesFromFeaturesToDataset(
    tf::OpKernelContext* ctx, dataset::VerticalDataset* dataset) {
  bool first_set_num_rows = true;
  const auto set_num_rows =
      [&first_set_num_rows, &dataset](
          const int64_t num_rows,
          const AbstractFeatureResource* feature) -> tf::Status {
    if (first_set_num_rows) {
      dataset->set_nrow(num_rows);
    } else if (dataset->nrow() != num_rows) {
      return tf::Status(
          tf::error::Code::INVALID_ARGUMENT,
          absl::Substitute(
              "Inconsistent number of observations "
              "between features for feature $0 != $1. For feature $2.",
              dataset->nrow(), num_rows, feature->feature_name()));
    }
    return tf::Status::OK();
  };

  TF_RETURN_IF_ERROR(IterateFeatures(
      [&](SimpleMLNumericalFeature::Resource* feature, const int feature_idx) {
        TF_RETURN_IF_ERROR(set_num_rows(feature->data().size(), feature));
        auto* col_data = dataset->MutableColumnWithCast<
            dataset::VerticalDataset::NumericalColumn>(feature_idx);
        *col_data->mutable_values() = std::move(*feature->mutable_data());
        feature->mutable_data()->clear();
        return tf::Status::OK();
      },
      [&](SimpleMLCategoricalStringFeature::Resource* feature,
          const int feature_idx) {
        TF_RETURN_IF_ERROR(
            set_num_rows(feature->indexed_data().size(), feature));
        const auto& col_spec = dataset->data_spec().columns(feature_idx);
        auto* col_data = dataset->MutableColumnWithCast<
            dataset::VerticalDataset::CategoricalColumn>(feature_idx);
        col_data->Resize(0);
        const auto& reverse_index = feature->reverse_index();
        for (const auto& indexed_value : feature->indexed_data()) {
          const auto& value = reverse_index[indexed_value];
          if (value.empty()) {
            col_data->AddNA();
          } else {
            col_data->Add(dataset::CategoricalStringToValue(value, col_spec));
          }
        }
        // Note: Thread annotations don't work in lambdas.
        feature->non_mutex_protected_clear();
        return tf::Status::OK();
      },
      [&](SimpleMLCategoricalIntFeature::Resource* feature,
          const int feature_idx) {
        TF_RETURN_IF_ERROR(set_num_rows(feature->data().size(), feature));
        const auto& col_spec = dataset->data_spec().columns(feature_idx);
        auto* col_data = dataset->MutableColumnWithCast<
            dataset::VerticalDataset::CategoricalColumn>(feature_idx);
        col_data->Resize(0);
        for (int value : feature->data()) {
          if (value < dataset::VerticalDataset::CategoricalColumn::kNaValue) {
            // Treated as missing value.
            value = dataset::VerticalDataset::CategoricalColumn::kNaValue;
          }
          if (value >= col_spec.categorical().number_of_unique_values()) {
            // Treated as out-of-dictionary.
            value = 0;
          }
          col_data->Add(value);
        }
        feature->mutable_data()->clear();
        return tf::Status::OK();
      },
      [&](SimpleMLCategoricalSetStringFeature::Resource* feature,
          const int feature_idx) {
        TF_RETURN_IF_ERROR(set_num_rows(feature->num_examples(), feature));
        const auto& col_spec = dataset->data_spec().columns(feature_idx);
        auto* col_data = dataset->MutableColumnWithCast<
            dataset::VerticalDataset::CategoricalSetColumn>(feature_idx);
        col_data->Resize(0);

        // Temporary buffer for the copy.
        std::vector<int> tmp_value;

        const int num_examples = feature->num_examples();
        for (int example_idx = 0; example_idx < num_examples; example_idx++) {
          // Get and convert the values.
          tmp_value.clear();
          const int begin_value_idx = feature->row_splits()[example_idx];
          const int end_value_idx = feature->row_splits()[example_idx + 1];
          for (int value_idx = begin_value_idx; value_idx < end_value_idx;
               value_idx++) {
            const auto& value_str = feature->values()[value_idx];
            const int32_t value =
                dataset::CategoricalStringToValue(value_str, col_spec);
            tmp_value.push_back(value);
          }

          // Store the values.
          std::sort(tmp_value.begin(), tmp_value.end());
          col_data->AddVector(tmp_value);
        }
        feature->non_mutex_protected_clear();
        return tf::Status::OK();
      },
      [&](SimpleMLCategoricalSetIntFeature::Resource* feature,
          const int feature_idx) {
        TF_RETURN_IF_ERROR(set_num_rows(feature->num_examples(), feature));
        const auto& col_spec = dataset->data_spec().columns(feature_idx);
        auto* col_data = dataset->MutableColumnWithCast<
            dataset::VerticalDataset::CategoricalSetColumn>(feature_idx);
        col_data->Resize(0);

        // Temporary buffer for the copy.
        std::vector<int> tmp_value;

        const int num_examples = feature->num_examples();
        for (int example_idx = 0; example_idx < num_examples; example_idx++) {
          // Get and check the values.
          tmp_value.clear();
          const int begin_value_idx = feature->row_splits()[example_idx];
          const int end_value_idx = feature->row_splits()[example_idx + 1];
          for (int value_idx = begin_value_idx; value_idx < end_value_idx;
               value_idx++) {
            if (value_idx < 0 || value_idx >= feature->values().size()) {
              return tf::Status(tf::error::Code::INTERNAL, "Internal error");
            }
            auto value = feature->values()[value_idx];
            if (value < dataset::VerticalDataset::CategoricalColumn::kNaValue) {
              return tf::Status(
                  tf::error::Code::INVALID_ARGUMENT,
                  absl::StrCat("Integer categorical value should "
                               "be >= -1. Found  value",
                               value, " for feature", feature->feature_name()));
            }
            if (value >= col_spec.categorical().number_of_unique_values()) {
              // Treated as out-of-dictionary.
              value = 0;
            }
            tmp_value.push_back(value);
          }

          // Store the values.
          std::sort(tmp_value.begin(), tmp_value.end());
          col_data->AddVector(tmp_value);
        }
        feature->non_mutex_protected_clear();
        return tf::Status::OK();
      },
      [&](SimpleMLHashFeature::Resource* feature, const int feature_idx) {
        TF_RETURN_IF_ERROR(set_num_rows(feature->data().size(), feature));
        auto* col_data =
            dataset
                ->MutableColumnWithCast<dataset::VerticalDataset::HashColumn>(
                    feature_idx);
        *col_data->mutable_values() = std::move(*feature->mutable_data());
        feature->mutable_data()->clear();
        return tf::Status::OK();
      }));

  return tf::Status::OK();
}

int FeatureSet::NumFeatures() const {
  return numerical_features_.size() + categorical_string_features_.size() +
         categorical_int_features_.size() +
         categorical_set_string_features_.size() +
         categorical_set_int_features_.size() + hash_features_.size();
}

// Trains a simpleML model.
class SimpleMLModelTrainer : public tensorflow::OpKernel {
 public:
  explicit SimpleMLModelTrainer(tf::OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_ids", &feature_ids_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("label_id", &label_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("weight_id", &weight_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("model_dir", &model_dir_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("model_id", &model_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("learner", &learner_));

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

    int task_idx;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("task", &task_idx));
    OP_REQUIRES(ctx, model::proto::Task_IsValid(task_idx),
                tf::Status(tf::error::INVALID_ARGUMENT, "Unknown task"));
    task_ = static_cast<model::proto::Task>(task_idx);

    {
      std::string serialized_training_config;
      OP_REQUIRES_OK(
          ctx, ctx->GetAttr("training_config", &serialized_training_config));
      if (!training_config_.MergeFromString(serialized_training_config)) {
        OP_REQUIRES_OK(
            ctx, tf::Status(tf::error::INVALID_ARGUMENT,
                            "Cannot de-serialize training_config proto."));
      }
      if (training_config_.has_task()) {
        OP_REQUIRES_OK(
            ctx,
            tf::Status(tf::error::INVALID_ARGUMENT,
                       "The \"task\" should not be set in the training_config,"
                       "instead set it as the Op parameter \"task\"."));
      }
      if (training_config_.has_learner()) {
        OP_REQUIRES_OK(
            ctx,
            tf::Status(
                tf::error::INVALID_ARGUMENT,
                "The \"learner\" should not be set in the training_config, "
                "instead set it as the Op parameter \"learner\"."));
      }
      if (training_config_.has_label()) {
        OP_REQUIRES_OK(
            ctx, tf::Status(
                     tf::error::INVALID_ARGUMENT,
                     "The \"label\" should not be set in the training_config, "
                     "instead set it as the Op parameter \"label_id\"."));
      }
      if (training_config_.has_weight_definition()) {
        OP_REQUIRES_OK(ctx,
                       tf::Status(tf::error::INVALID_ARGUMENT,
                                  "The \"weight_definition\" should not be "
                                  "set in the training_config."));
      }
      if (training_config_.features_size() > 0) {
        OP_REQUIRES_OK(
            ctx,
            tf::Status(
                tf::error::INVALID_ARGUMENT,
                "The \"features\" should not be set in the training_config, "
                "for this Op they are generated automatically."));
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

  ~SimpleMLModelTrainer() override = default;

  void Compute(tf::OpKernelContext* ctx) override {
    LOG(INFO) << "Start Yggdrasil model training";
    LOG(INFO) << "Collect training examples";

    tf::Tensor* success_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, tf::TensorShape({}), &success_tensor));
    auto success = success_tensor->scalar<bool>();
    success() = true;

    if (!HasTrainingExamples(ctx)) {
      LOG(INFO) << "Not training example available. Ignore training request.";
      success() = false;
      return;
    }

    dataset::VerticalDataset dataset;
    std::string label_feature;
    std::string weight_feature;
    std::vector<std::string> input_features;
    OP_REQUIRES_OK(ctx, CreateTrainingDatasetFromFeatures(
                            ctx, &dataset, &label_feature, &weight_feature,
                            &input_features));

    LOG(INFO) << "Dataset:\n"
              << dataset::PrintHumanReadable(dataset.data_spec(), false);

    LOG(INFO) << "Configure learner";
    model::proto::TrainingConfig config = training_config_;
    config.set_learner(learner_);
    config.set_label(label_feature);
    config.set_task(task_);
    if (!weight_feature.empty()) {
      LOG(INFO) << "Use example weight: " << weight_feature
                << " from accumulator: " << weight_id_;
      config.mutable_weight_definition()->set_attribute(weight_feature);
      config.mutable_weight_definition()->mutable_numerical();
    }
    for (const auto& input_feature : input_features) {
      config.add_features(
          dataset::EscapeTrainingConfigFeatureName(input_feature));
    }

    std::unique_ptr<model::AbstractLearner> learner;
    OP_REQUIRES_OK(ctx, utils::FromUtilStatus(GetLearner(config, &learner)));
    OP_REQUIRES_OK(
        ctx, utils::FromUtilStatus(learner->SetHyperParameters(hparams_)));
    *learner->mutable_deployment() = deployment_config_;
    if (!model_dir_.empty()) {
      learner->set_log_directory(tf::io::JoinPath(model_dir_, "train_logs"));
    }

    LOG(INFO) << "Training config:\n"
              << learner->training_config().DebugString();

    LOG(INFO) << "Deployment config:\n" << learner->deployment().DebugString();

    // The following commented code snippet exports the dataset and training
    // configuration so it can be run easily in a debugger by running:
    //
    // bazel run -c opt //third_party/yggdrasil_decision_forests/cli:train -- \
    //   --alsologtostderr --output=/tmp/model \
    //   --dataset=tfrecord+tfe:/tmp/dataset.tfe \
    //   --dataspec=/tmp/dataspec.pbtxt \
    //   --config=/tmp/train_config.pbtxt
    //
    // Add the dependency:
    //   //third_party/yggdrasil_decision_forests/dataset:tf_example_io_tfrecord
    //
    /*
    CHECK_OK(SaveVerticalDataset(dataset, "tfrecord+tfe:/tmp/dataset.tfe"));
    CHECK_OK(file::SetTextProto("/tmp/dataspec.pbtxt", dataset.data_spec(),
                                file::Defaults()));
    CHECK_OK(file::SetTextProto("/tmp/train_config.pbtxt",
                                learner->training_config(), file::Defaults()));
    */

    LOG(INFO) << "Train model";
    auto model = learner->TrainWithStatus(dataset);
    OP_REQUIRES_OK(ctx, utils::FromUtilStatus(model.status()));

    // Export model to disk.
    if (!model_dir_.empty()) {
      LOG(INFO) << "Export model in log directory: " << model_dir_;
      OP_REQUIRES_OK(ctx, utils::FromUtilStatus(
                              SaveModel(tf::io::JoinPath(model_dir_, "model"),
                                        model.value().get())));
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
  tf::Status CreateTrainingDatasetFromFeatures(
      tf::OpKernelContext* ctx, dataset::VerticalDataset* dataset,
      std::string* label_feature, std::string* weight_feature,
      std::vector<std::string>* input_features) {
    FeatureSet feature_set;
    TF_RETURN_IF_ERROR(
        feature_set.Link(ctx, feature_ids_, label_id_, weight_id_, nullptr));
    TF_RETURN_IF_ERROR(
        feature_set.InitializeDatasetFromFeatures(ctx, guide_, dataset));
    TF_RETURN_IF_ERROR(
        feature_set.MoveExamplesFromFeaturesToDataset(ctx, dataset));
    *label_feature = feature_set.label_feature();
    *weight_feature = feature_set.weight_feature();
    *input_features = feature_set.input_features();
    return tf::Status::OK();
  }

  bool HasTrainingExamples(tf::OpKernelContext* ctx) {
    // Note: The resource manager container is created when the first batch of
    // training examples are consumed.
    AbstractFeatureResource* tmp_feature;
    const auto label_status =
        ctx->resource_manager()->Lookup<AbstractFeatureResource, true>(
            kModelContainer, label_id_, &tmp_feature);
    return label_status.ok();
  }

  std::string feature_ids_;
  std::string label_id_;
  std::string weight_id_;
  std::string model_dir_;
  std::string model_id_;
  std::string learner_;

  model::proto::GenericHyperParameters hparams_;
  model::proto::Task task_;
  model::proto::TrainingConfig training_config_;
  model::proto::DeploymentConfig deployment_config_;
  dataset::proto::DataSpecificationGuide guide_;
};

REGISTER_KERNEL_BUILDER(Name("SimpleMLModelTrainer").Device(tf::DEVICE_CPU),
                        SimpleMLModelTrainer);

// Utility class for operations on simpleML models stored in a resource.
class AbstractSimpleMLModelOp : public tensorflow::OpKernel {
 public:
  explicit AbstractSimpleMLModelOp(tf::OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("model_identifier", &model_id_));
  }

  ~AbstractSimpleMLModelOp() override = default;

  // Called when the op is applied. If "model"==nullptr, the model is not
  // available.
  virtual void ComputeModel(tf::OpKernelContext* ctx,
                            const model::AbstractModel* const model) = 0;

  void Compute(tf::OpKernelContext* ctx) override {
    YggdrasilModelContainer* model_container;
    const auto lookup_status = ctx->resource_manager()->Lookup(
        kModelContainer, model_id_, &model_container);
    if (!lookup_status.ok()) {
      ComputeModel(ctx, nullptr);
      return;
    }
    ComputeModel(ctx, model_container->mutable_model()->get());
    model_container->Unref();
  }

 private:
  std::string model_id_;
};

// Build a text description of the model.
class SimpleMLShowModel : public AbstractSimpleMLModelOp {
 public:
  explicit SimpleMLShowModel(tf::OpKernelConstruction* ctx)
      : AbstractSimpleMLModelOp(ctx) {}

  void ComputeModel(tf::OpKernelContext* ctx,
                    const model::AbstractModel* const model) override {
    tf::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, tf::TensorShape({}), &output_tensor));
    auto output = output_tensor->scalar<tensorflow::tstring>();
    if (!model) {
      output().clear();
    }

    std::string description;
    model->AppendDescriptionAndStatistics(/*full_definition=*/false,
                                          &description);
    output() = description;
  }
};

REGISTER_KERNEL_BUILDER(Name("SimpleMLShowModel").Device(tf::DEVICE_CPU),
                        SimpleMLShowModel);

// Unload a model from memory.
class SimpleMLUnloadModel : public tf::OpKernel {
 public:
  explicit SimpleMLUnloadModel(tf::OpKernelConstruction* ctx)
      : tf::OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("model_identifier", &model_id_));
  }

  void Compute(tf::OpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx,
                   ctx->resource_manager()->Delete<YggdrasilModelContainer>(
                       kModelContainer, model_id_));
  }

 private:
  std::string model_id_;
};

REGISTER_KERNEL_BUILDER(Name("SimpleMLUnloadModel").Device(tf::DEVICE_CPU),
                        SimpleMLUnloadModel);

}  // namespace ops
}  // namespace tensorflow_decision_forests
