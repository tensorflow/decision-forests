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
#include <csignal>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/types/optional.h"
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
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/tensorflow.h"

namespace tensorflow_decision_forests {
namespace ops {
namespace {
namespace tf = ::tensorflow;
namespace ydf = ::yggdrasil_decision_forests;
namespace model = ydf::model;
namespace utils = ydf::utils;
namespace dataset = ydf::dataset;

}  // namespace

absl::Status YggdrasilModelContainer::LoadModel(
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
  return absl::OkStatus();
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
    if (model_paths.size() != 1) {
      OP_REQUIRES_OK(ctx, absl::Status(static_cast<absl::StatusCode>(
                                           absl::StatusCode::kInvalidArgument),
                                       "Wrong number of models"));
    }
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

absl::Status FeatureSet::Link(
    tf::OpKernelContext* ctx, const std::vector<std::string>& resource_ids,
    const dataset::proto::DataSpecification* const existing_dataspec,
    const DatasetType dataset_type) {
  std::vector<std::string> sorted_resource_ids = resource_ids;
  std::sort(sorted_resource_ids.begin(), sorted_resource_ids.end());

  for (const auto& column_id : sorted_resource_ids) {
    std::string resource_id = column_id;
    switch (dataset_type) {
      case DatasetType::kTraining:
        break;
      case DatasetType::kValidation:
        // See "_FEATURE_RESOURCE_VALIDATION_SUFFIX" in "core.py".
        absl::StrAppend(&resource_id, "__VALIDATION");
        break;
    }

    AbstractFeatureResource* feature;
    TF_RETURN_IF_ERROR(
        ctx->resource_manager()->Lookup<AbstractFeatureResource, true>(
            kModelContainer, resource_id, &feature));

    const int feature_idx =
        existing_dataspec ? dataset::GetColumnIdxFromName(
                                feature->feature_name(), *existing_dataspec)
                          : NumFeatures();

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
      return absl::Status(
          static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
          absl::StrCat("Unsupported type for feature \"",
                       feature->feature_name(), "\""));
    }
  }

  return absl::OkStatus();
}

absl::Status FeatureSet::IterateFeatures(
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

  return absl::OkStatus();
}

absl::Status FeatureSet::Unlink() {
  TF_RETURN_IF_ERROR(IterateFeatures(
      [](SimpleMLNumericalFeature::Resource* feature, const int feature_idx) {
        feature->Unref();
        return absl::OkStatus();
      },
      [](SimpleMLCategoricalStringFeature::Resource* feature,
         const int feature_idx) {
        feature->Unref();
        return absl::OkStatus();
      },
      [](SimpleMLCategoricalIntFeature::Resource* feature,
         const int feature_idx) {
        feature->Unref();
        return absl::OkStatus();
      },
      [](SimpleMLCategoricalSetStringFeature::Resource* feature,
         const int feature_idx) {
        feature->Unref();
        return absl::OkStatus();
      },
      [](SimpleMLCategoricalSetIntFeature::Resource* feature,
         const int feature_idx) {
        feature->Unref();
        return absl::OkStatus();
      },
      [](SimpleMLHashFeature::Resource* feature, const int feature_idx) {
        feature->Unref();
        return absl::OkStatus();
      }));
  numerical_features_.clear();
  categorical_string_features_.clear();
  categorical_int_features_.clear();
  categorical_set_string_features_.clear();
  categorical_set_int_features_.clear();
  hash_features_.clear();
  return absl::OkStatus();
}

// Initialize a dataset (including the dataset's dataspec) from the linked
// resource aggregators.
absl::Status FeatureSet::InitializeDatasetFromFeatures(
    tf::OpKernelContext* ctx,
    const dataset::proto::DataSpecificationGuide& guide,
    dataset::VerticalDataset* dataset) {
  int64_t num_batches = -1;
  int64_t num_examples = -1;
  const auto set_num_examples =
      [&num_examples, &num_batches](
          const int64_t observed_num_examples,
          const int64_t observed_num_batches) -> absl::Status {
    if (num_examples == -1) {
      num_examples = observed_num_examples;
      num_batches = observed_num_batches;
      return absl::OkStatus();
    }
    if (num_examples != observed_num_examples) {
      return absl::Status(
          static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
          absl::Substitute("Inconsistent number of training examples for the "
                           "different input features $0 != $1.",
                           num_examples, observed_num_examples));
    }
    return absl::OkStatus();
  };

  for (int feature_idx = 0; feature_idx < NumFeatures(); feature_idx++) {
    dataset->mutable_data_spec()->add_columns();
  }

  // Apply the guide on a column. The type of the column should be set.
  const auto apply_guide = [&](const absl::string_view feature_name,
                               dataset::proto::Column* col,
                               const bool apply_type = false) -> absl::Status {
    dataset::proto::ColumnGuide col_guide;
    TF_RETURN_IF_ERROR_FROM_ABSL_STATUS(
        dataset::BuildColumnGuide(feature_name, guide, &col_guide).status());
    if (apply_type) {
      if (col_guide.has_type()) {
        col->set_type(col_guide.type());
      } else {
        if (col->type() == dataset::proto::NUMERICAL &&
            guide.detect_numerical_as_discretized_numerical()) {
          col->set_type(dataset::proto::DISCRETIZED_NUMERICAL);
        }
      }
    }
    return utils::FromUtilStatus(
        dataset::UpdateSingleColSpecWithGuideInfo(col_guide, col));
  };

  TF_RETURN_IF_ERROR(IterateFeatures(
      [&](SimpleMLNumericalFeature::Resource* feature, const int feature_idx) {
        auto* col = dataset->mutable_data_spec()->mutable_columns(feature_idx);
        col->set_name(feature->feature_name());
        col->set_type(dataset::proto::ColumnType::NUMERICAL);
        TF_RETURN_IF_ERROR(
            apply_guide(feature->feature_name(), col, /*apply_type=*/true));
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
        return absl::OkStatus();
      },
      [&](SimpleMLCategoricalIntFeature::Resource* feature,
          const int feature_idx) {
        auto* col = dataset->mutable_data_spec()->mutable_columns(feature_idx);
        col->set_name(feature->feature_name());
        col->set_type(dataset::proto::ColumnType::CATEGORICAL);
        TF_RETURN_IF_ERROR(apply_guide(feature->feature_name(), col));
        // Both in TF-DF and SimpleML Estimator, integer values are offset by 1.
        // See CATEGORICAL_INTEGER_OFFSET.
        col->mutable_categorical()->set_offset_value_by_one_during_training(
            true);
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
    return absl::Status(
        static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
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

        // Is the numerical column discretized?
        const bool discretized =
            col->type() == dataset::proto::ColumnType::DISCRETIZED_NUMERICAL;

        for (const auto value : feature->data()) {
          TF_RETURN_IF_ERROR_FROM_ABSL_STATUS(
              dataset::UpdateNumericalColumnSpec(value, col, col_acc));
          if (discretized) {
            dataset::UpdateComputeSpecDiscretizedNumerical(value, col, col_acc);
          }
        }
        return absl::OkStatus();
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
        return absl::OkStatus();
      },
      [&](SimpleMLCategoricalIntFeature::Resource* feature,
          const int feature_idx) {
        auto* col = dataset->mutable_data_spec()->mutable_columns(feature_idx);
        auto* col_acc = accumulator.mutable_columns(feature_idx);
        for (const auto value : feature->data()) {
          TF_RETURN_IF_ERROR_FROM_ABSL_STATUS(
              dataset::UpdateCategoricalIntColumnSpec(value, col, col_acc));
        }
        return absl::OkStatus();
      },
      [&](SimpleMLCategoricalSetStringFeature::Resource* feature,
          const int feature_idx) {
        auto* col = dataset->mutable_data_spec()->mutable_columns(feature_idx);
        auto* col_acc = accumulator.mutable_columns(feature_idx);
        for (const auto& value : feature->values()) {
          TF_RETURN_IF_ERROR_FROM_ABSL_STATUS(
              dataset::UpdateCategoricalStringColumnSpec(value, col, col_acc));
        }
        return absl::OkStatus();
      },
      [&](SimpleMLCategoricalSetIntFeature::Resource* feature,
          const int feature_idx) {
        auto* col = dataset->mutable_data_spec()->mutable_columns(feature_idx);
        auto* col_acc = accumulator.mutable_columns(feature_idx);
        for (const auto value : feature->values()) {
          TF_RETURN_IF_ERROR_FROM_ABSL_STATUS(
              dataset::UpdateCategoricalIntColumnSpec(value, col, col_acc));
        }
        return absl::OkStatus();
      },
      [&](SimpleMLHashFeature::Resource* feature, const int feature_idx) {
        // Nothing to do.
        return absl::OkStatus();
      }));

  TF_RETURN_IF_ERROR_FROM_ABSL_STATUS(dataset::FinalizeComputeSpec(
      guide, accumulator, dataset->mutable_data_spec()));

  return absl::OkStatus();
}

absl::Status FeatureSet::MoveExamplesFromFeaturesToDataset(
    tf::OpKernelContext* ctx, dataset::VerticalDataset* dataset) {
  bool first_set_num_rows = true;
  const auto set_num_rows =
      [&first_set_num_rows, &dataset](
          const int64_t num_rows,
          const AbstractFeatureResource* feature) -> absl::Status {
    if (first_set_num_rows) {
      dataset->set_nrow(num_rows);
    } else if (dataset->nrow() != num_rows) {
      return absl::Status(
          static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
          absl::Substitute(
              "Inconsistent number of observations "
              "between features for feature $0 != $1. For feature $2.",
              dataset->nrow(), num_rows, feature->feature_name()));
    }
    return absl::OkStatus();
  };

  TF_RETURN_IF_ERROR(IterateFeatures(
      [&](SimpleMLNumericalFeature::Resource* feature,
          const int feature_idx) -> absl::Status {
        TF_RETURN_IF_ERROR(set_num_rows(feature->data().size(), feature));
        const auto& col = dataset->mutable_data_spec()->columns(feature_idx);

        // Is the numerical column discretized?
        const bool discretized =
            col.type() == dataset::proto::ColumnType::DISCRETIZED_NUMERICAL;
        if (discretized) {
          // Copy the discretized numerical values.
          TF_ASSIGN_OR_RETURN_FROM_ABSL_STATUS(
              auto* col_data,
              dataset->MutableColumnWithCastWithStatus<
                  dataset::VerticalDataset::DiscretizedNumericalColumn>(
                  feature_idx));
          col_data->Resize(0);
          for (float value : feature->data()) {
            col_data->Add(dataset::NumericalToDiscretizedNumerical(col, value));
          }
        } else {
          // Copy the non discretized values.
          TF_ASSIGN_OR_RETURN_FROM_ABSL_STATUS(
              auto* col_data,
              dataset->MutableColumnWithCastWithStatus<
                  dataset::VerticalDataset::NumericalColumn>(feature_idx));
          *col_data->mutable_values() = std::move(*feature->mutable_data());
        }
        feature->mutable_data()->clear();
        return absl::OkStatus();
      },
      [&](SimpleMLCategoricalStringFeature::Resource* feature,
          const int feature_idx) -> absl::Status {
        TF_RETURN_IF_ERROR(
            set_num_rows(feature->indexed_data().size(), feature));
        const auto& col_spec = dataset->data_spec().columns(feature_idx);
        TF_ASSIGN_OR_RETURN_FROM_ABSL_STATUS(
            auto* col_data,
            dataset->MutableColumnWithCastWithStatus<
                dataset::VerticalDataset::CategoricalColumn>(feature_idx));
        col_data->Resize(0);
        const auto& reverse_index = feature->reverse_index();
        for (const auto& indexed_value : feature->indexed_data()) {
          const auto& value = reverse_index[indexed_value];
          if (value.empty()) {
            col_data->AddNA();
          } else {
            TF_ASSIGN_OR_RETURN_FROM_ABSL_STATUS(
                auto int_value,
                dataset::CategoricalStringToValueWithStatus(value, col_spec));
            col_data->Add(int_value);
          }
        }
        // Note: Thread annotations don't work in lambdas.
        feature->non_mutex_protected_clear();
        return absl::OkStatus();
      },
      [&](SimpleMLCategoricalIntFeature::Resource* feature,
          const int feature_idx) -> absl::Status {
        TF_RETURN_IF_ERROR(set_num_rows(feature->data().size(), feature));
        const auto& col_spec = dataset->data_spec().columns(feature_idx);
        TF_ASSIGN_OR_RETURN_FROM_ABSL_STATUS(
            auto* col_data,
            dataset->MutableColumnWithCastWithStatus<
                dataset::VerticalDataset::CategoricalColumn>(feature_idx));
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
        return absl::OkStatus();
      },
      [&](SimpleMLCategoricalSetStringFeature::Resource* feature,
          const int feature_idx) -> absl::Status {
        TF_RETURN_IF_ERROR(set_num_rows(feature->num_examples(), feature));
        const auto& col_spec = dataset->data_spec().columns(feature_idx);
        TF_ASSIGN_OR_RETURN_FROM_ABSL_STATUS(
            auto* col_data,
            dataset->MutableColumnWithCastWithStatus<
                dataset::VerticalDataset::CategoricalSetColumn>(feature_idx));
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
            TF_ASSIGN_OR_RETURN_FROM_ABSL_STATUS(
                const int32_t value,
                dataset::CategoricalStringToValueWithStatus(value_str,
                                                            col_spec));
            tmp_value.push_back(value);
          }

          // Store the values.
          std::sort(tmp_value.begin(), tmp_value.end());
          tmp_value.erase(std::unique(tmp_value.begin(), tmp_value.end()),
                          tmp_value.end());

          col_data->AddVector(tmp_value);
        }
        feature->non_mutex_protected_clear();
        return absl::OkStatus();
      },
      [&](SimpleMLCategoricalSetIntFeature::Resource* feature,
          const int feature_idx) -> absl::Status {
        TF_RETURN_IF_ERROR(set_num_rows(feature->num_examples(), feature));
        const auto& col_spec = dataset->data_spec().columns(feature_idx);
        TF_ASSIGN_OR_RETURN_FROM_ABSL_STATUS(
            auto* col_data,
            dataset->MutableColumnWithCastWithStatus<
                dataset::VerticalDataset::CategoricalSetColumn>(feature_idx));
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
              return absl::Status(
                  static_cast<absl::StatusCode>(absl::StatusCode::kInternal),
                  "Internal error");
            }
            auto value = feature->values()[value_idx];
            if (value < dataset::VerticalDataset::CategoricalColumn::kNaValue) {
              return absl::Status(
                  static_cast<absl::StatusCode>(
                      absl::StatusCode::kInvalidArgument),
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
          tmp_value.erase(std::unique(tmp_value.begin(), tmp_value.end()),
                          tmp_value.end());

          col_data->AddVector(tmp_value);
        }
        feature->non_mutex_protected_clear();
        return absl::OkStatus();
      },
      [&](SimpleMLHashFeature::Resource* feature,
          const int feature_idx) -> absl::Status {
        TF_RETURN_IF_ERROR(set_num_rows(feature->data().size(), feature));
        TF_ASSIGN_OR_RETURN_FROM_ABSL_STATUS(
            auto* col_data,
            dataset->MutableColumnWithCastWithStatus<
                dataset::VerticalDataset::HashColumn>(feature_idx));
        *col_data->mutable_values() = std::move(*feature->mutable_data());
        feature->mutable_data()->clear();
        return absl::OkStatus();
      }));

  return absl::OkStatus();
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
    OP_REQUIRES_OK(ctx, ctx->GetAttr("resource_ids", &resource_ids_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("model_dir", &model_dir_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("model_id", &model_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_file_prefix", &use_file_prefix_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("create_model_resource", &create_model_resource_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blocking", &blocking_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("node_format", &node_format_));

    if (model_id_.empty()) {
      OP_REQUIRES_OK(ctx, absl::Status(static_cast<absl::StatusCode>(
                                           absl::StatusCode::kInvalidArgument),
                                       "Model id is empty"));
    }

    std::string serialized_guide;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("guide", &serialized_guide));
    if (!guide_.ParseFromString(serialized_guide)) {
      OP_REQUIRES_OK(ctx, absl::Status(static_cast<absl::StatusCode>(
                                           absl::StatusCode::kInvalidArgument),
                                       "Cannot de-serialize guide proto."));
    }

    std::string hparams;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("hparams", &hparams));
    if (!hparams_.ParseFromString(hparams)) {
      OP_REQUIRES_OK(ctx, absl::Status(static_cast<absl::StatusCode>(
                                           absl::StatusCode::kInvalidArgument),
                                       "Cannot de-serialize hparams proto."));
    }
    {
      std::string serialized_training_config;
      OP_REQUIRES_OK(
          ctx, ctx->GetAttr("training_config", &serialized_training_config));
      if (!training_config_.MergeFromString(serialized_training_config)) {
        OP_REQUIRES_OK(
            ctx, absl::Status(static_cast<absl::StatusCode>(
                                  absl::StatusCode::kInvalidArgument),
                              "Cannot de-serialize training_config proto."));
      }
      if (!training_config_.has_task()) {
        OP_REQUIRES_OK(ctx,
                       absl::Status(static_cast<absl::StatusCode>(
                                        absl::StatusCode::kInvalidArgument),
                                    "\"task\" not set"));
      }
      if (!training_config_.has_learner()) {
        OP_REQUIRES_OK(ctx,
                       absl::Status(static_cast<absl::StatusCode>(
                                        absl::StatusCode::kInvalidArgument),
                                    "\"learner\" not set"));
      }
      if (!training_config_.has_label()) {
        OP_REQUIRES_OK(ctx,
                       absl::Status(static_cast<absl::StatusCode>(
                                        absl::StatusCode::kInvalidArgument),
                                    "\"label\" not set"));
      }
    }

    {
      std::string serialized_deployment_config;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("deployment_config",
                                       &serialized_deployment_config));
      if (!deployment_config_.MergeFromString(serialized_deployment_config)) {
        OP_REQUIRES_OK(
            ctx, absl::Status(static_cast<absl::StatusCode>(
                                  absl::StatusCode::kInvalidArgument),
                              "Cannot de-serialize deployment_config proto."));
      }
    }

    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("has_validation_dataset", &has_validation_dataset_));
  }

  ~SimpleMLModelTrainer() override = default;

  void Compute(tf::OpKernelContext* ctx) override {
    LOG(INFO) << "Start Yggdrasil model training";
    LOG(INFO) << "Collect training examples";

    tf::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, tf::TensorShape({}), &output_tensor));
    output_tensor->scalar<int32_t>()() = -1;

    if (!HasTrainingExamples(ctx)) {
      LOG(WARNING) << "No training example available. Ignore training request.";
      return;
    }

    LOG(INFO) << "Dataspec guide:\n" << guide_.DebugString();

    auto dataset = absl::make_unique<dataset::VerticalDataset>();
    OP_REQUIRES_OK(
        ctx, BuildVerticalDatasetFromTFResources(ctx, DatasetType::kTraining,
                                                 resource_ids_, dataset.get()));

    LOG(INFO) << "Training dataset:\n"
              << dataset::PrintHumanReadable(dataset->data_spec(), false);

    std::unique_ptr<dataset::VerticalDataset> valid_dataset;
    if (has_validation_dataset_) {
      LOG(INFO) << "Collect validation dataset";
      valid_dataset = absl::make_unique<dataset::VerticalDataset>();
      OP_REQUIRES_OK(ctx, BuildVerticalDatasetFromTFResources(
                              ctx, DatasetType::kValidation, resource_ids_,
                              valid_dataset.get()));

      LOG(INFO) << "Validation dataset:\n"
                << dataset::PrintHumanReadable(valid_dataset->data_spec(),
                                               false);
    }

    LOG(INFO) << "Configure learner";
    model::proto::TrainingConfig config = training_config_;

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
    // bazel run -c opt //third_party/yggdrasil_decision_forests/cli:train --
    // \
    //   --alsologtostderr --output=/tmp/model \
    //   --dataset=tfrecord+tfe:/tmp/dataset.tfe \
    //   --dataspec=/tmp/dataspec.pbtxt \
    //   --config=/tmp/train_config.pbtxt
    //
    // Add the dependency:
    //   //third_party/yggdrasil_decision_forests/dataset/tensorflow:tf_example_io_tfrecord
    //
    /*
    CHECK_OK(SaveVerticalDataset(dataset, "tfrecord+tfe:/tmp/dataset.tfe"));
    CHECK_OK(file::SetTextProto("/tmp/dataspec.pbtxt", dataset.data_spec(),
                                file::Defaults()));
    CHECK_OK(file::SetTextProto("/tmp/train_config.pbtxt",
                                learner->training_config(),
    file::Defaults()));
    */

#ifdef TFDF_STOP_TRAINING_ON_INTERRUPT
    OP_REQUIRES_OK(ctx, interruption::EnableUserInterruption());
    learner->set_stop_training_trigger(&interruption::stop_training);
#endif

    YggdrasilModelContainer* model_container = nullptr;
    if (create_model_resource_) {
      model_container = new YggdrasilModelContainer();
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Create(
                              kModelContainer, model_id_, model_container));
    }

    // Create a std::function to train the model.
    //
    // Note: The capture of std::function should be copiable.
    struct TrainingState {
      std::string model_dir;
      bool use_file_prefix;
      std::string model_id;
      YggdrasilModelContainer* model_container;
      std::unique_ptr<dataset::VerticalDataset> valid_dataset;
      std::unique_ptr<dataset::VerticalDataset> dataset;
      std::unique_ptr<model::AbstractLearner> learner;
      std::string node_format;
    };

    auto training_state = std::make_shared<TrainingState>();
    training_state->model_dir = this->model_dir_;
    training_state->model_id = this->model_id_;
    training_state->use_file_prefix = this->use_file_prefix_;
    training_state->model_container = model_container;
    training_state->valid_dataset = std::move(valid_dataset);
    training_state->dataset = std::move(dataset);
    training_state->learner = std::move(learner);
    training_state->node_format = this->node_format_;

    auto async_train = [training_state]() -> absl::Status {
      LOG(INFO) << "Train model";
      absl::StatusOr<std::unique_ptr<model::AbstractModel>> model;
      if (training_state->valid_dataset) {
        model = training_state->learner->TrainWithStatus(
            *training_state->dataset, *training_state->valid_dataset);
      } else {
        model =
            training_state->learner->TrainWithStatus(*training_state->dataset);
      }

#ifdef TFDF_STOP_TRAINING_ON_INTERRUPT
      RETURN_IF_ERROR(
          utils::ToUtilStatus(interruption::DisableUserInterruption()));
#endif

      RETURN_IF_ERROR(model.status());

      // If the model is a decision forest, set the node format.
      if (!training_state->node_format.empty()) {
        // Set the model format.
        auto* df_model =
            dynamic_cast<model::DecisionForestInterface*>(model.value().get());
        if (df_model) {
          df_model->set_node_format(training_state->node_format);
        } else {
          LOG(INFO) << "The node format cannot be set for this model type";
        }
      }

      // Export model to disk.
      if (!training_state->model_dir.empty()) {
        if (training_state->use_file_prefix) {
          LOG(INFO) << "Export model in log directory: "
                    << training_state->model_dir << " with prefix "
                    << training_state->model_id;
          RETURN_IF_ERROR(
              SaveModel(tf::io::JoinPath(training_state->model_dir, "model"),
                        model.value().get(),
                        {/*.file_prefix =*/training_state->model_id}));
        } else {
          LOG(INFO) << "Export model in log directory: "
                    << training_state->model_dir << " without prefix";
          RETURN_IF_ERROR(
              SaveModel(tf::io::JoinPath(training_state->model_dir, "model"),
                        model.value().get()));
        }
      }

      // Export model to model resource.
      if (training_state->model_container) {
        LOG(INFO) << "Save model in resources";
        *training_state->model_container->mutable_model() =
            std::move(model.value());
      }

      return absl::OkStatus();
    };

    auto process_id_or = StartLongRunningProcess(ctx, std::move(async_train));
    if (!process_id_or.ok()) {
      OP_REQUIRES_OK(ctx, utils::FromUtilStatus(process_id_or.status()));
    }
    output_tensor->scalar<int32_t>()() = process_id_or.value();

    if (blocking_) {
      while (true) {
        auto status_or =
            GetLongRunningProcessStatus(ctx, process_id_or.value());
        if (!status_or.ok()) {
          OP_REQUIRES_OK(ctx, utils::FromUtilStatus(status_or.status()));
        }
        if (status_or.value() == LongRunningProcessStatus::kSuccess) {
          break;
        }
      }
    }
  }

 private:
  absl::Status BuildVerticalDatasetFromTFResources(
      tf::OpKernelContext* ctx, const DatasetType dataset_type,
      const std::vector<std::string>& resource_ids,
      dataset::VerticalDataset* dataset) {
    FeatureSet feature_set;
    TF_RETURN_IF_ERROR(
        feature_set.Link(ctx, resource_ids, nullptr, dataset_type));
    TF_RETURN_IF_ERROR(
        feature_set.InitializeDatasetFromFeatures(ctx, guide_, dataset));
    TF_RETURN_IF_ERROR(
        feature_set.MoveExamplesFromFeaturesToDataset(ctx, dataset));
    return absl::OkStatus();
  }

  bool HasTrainingExamples(tf::OpKernelContext* ctx) {
    // Note: The resource manager container is created when the first batch of
    // training examples are consumed.
    if (resource_ids_.empty()) {
      return false;
    }
    AbstractFeatureResource* tmp_feature;
    const auto label_status =
        ctx->resource_manager()->Lookup<AbstractFeatureResource, true>(
            kModelContainer, resource_ids_.front(), &tmp_feature);
    tmp_feature->Unref();
    return label_status.ok();
  }

  std::vector<std::string> resource_ids_;

  std::string model_dir_;
  std::string model_id_;
  bool create_model_resource_;
  bool use_file_prefix_;
  bool blocking_;
  std::string node_format_;

  model::proto::GenericHyperParameters hparams_;
  model::proto::TrainingConfig training_config_;
  model::proto::DeploymentConfig deployment_config_;
  dataset::proto::DataSpecificationGuide guide_;
  bool has_validation_dataset_;
};

REGISTER_KERNEL_BUILDER(Name("SimpleMLModelTrainer").Device(tf::DEVICE_CPU),
                        SimpleMLModelTrainer);

class SimpleMLCheckTrainingConfiguration : public tensorflow::OpKernel {
 public:
  explicit SimpleMLCheckTrainingConfiguration(tf::OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    std::string hparams;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("hparams", &hparams));
    if (!hparams_.ParseFromString(hparams)) {
      OP_REQUIRES_OK(ctx, absl::Status(static_cast<absl::StatusCode>(
                                           absl::StatusCode::kInvalidArgument),
                                       "Cannot de-serialize hparams proto."));
    }

    {
      std::string serialized_training_config;
      OP_REQUIRES_OK(
          ctx, ctx->GetAttr("training_config", &serialized_training_config));
      if (!training_config_.MergeFromString(serialized_training_config)) {
        OP_REQUIRES_OK(
            ctx, absl::Status(static_cast<absl::StatusCode>(
                                  absl::StatusCode::kInvalidArgument),
                              "Cannot de-serialize training_config proto."));
      }
    }
  }

  ~SimpleMLCheckTrainingConfiguration() override = default;

  void Compute(tf::OpKernelContext* ctx) override {
    if (!training_config_.has_task()) {
      OP_REQUIRES_OK(ctx, absl::Status(static_cast<absl::StatusCode>(
                                           absl::StatusCode::kInvalidArgument),
                                       "\"task\" not set"));
    }
    if (!training_config_.has_learner()) {
      OP_REQUIRES_OK(ctx, absl::Status(static_cast<absl::StatusCode>(
                                           absl::StatusCode::kInvalidArgument),
                                       "\"learner\" not set"));
    }
    if (!training_config_.has_label()) {
      OP_REQUIRES_OK(ctx, absl::Status(static_cast<absl::StatusCode>(
                                           absl::StatusCode::kInvalidArgument),
                                       "\"label\" not set"));
    }

    // Check the parameters by creating a learner.
    std::unique_ptr<model::AbstractLearner> learner;
    OP_REQUIRES_OK(
        ctx, utils::FromUtilStatus(GetLearner(training_config_, &learner)));
    OP_REQUIRES_OK(
        ctx, utils::FromUtilStatus(learner->SetHyperParameters(hparams_)));
  }

  model::proto::GenericHyperParameters hparams_;
  model::proto::TrainingConfig training_config_;
};

REGISTER_KERNEL_BUILDER(
    Name("SimpleMLCheckTrainingConfiguration").Device(tf::DEVICE_CPU),
    SimpleMLCheckTrainingConfiguration);

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
    if (!model) {
      OP_REQUIRES_OK(ctx, absl::Status(static_cast<absl::StatusCode>(
                                           absl::StatusCode::kInvalidArgument),
                                       "The model does not exist."));
    }

    tf::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, tf::TensorShape({}), &output_tensor));
    auto output = output_tensor->scalar<tensorflow::tstring>();
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

// Sets the amount of logging.
class YggdrasilDecisionForestsSetLoggingLevel : public tf::OpKernel {
 public:
  explicit YggdrasilDecisionForestsSetLoggingLevel(
      tf::OpKernelConstruction* ctx)
      : tf::OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("level", &level_));
  }

  void Compute(tf::OpKernelContext* ctx) override {
    ydf::logging::SetLoggingLevel(level_);
  }

 private:
  int level_;
};

REGISTER_KERNEL_BUILDER(
    Name("YggdrasilDecisionForestsSetLoggingLevel").Device(tf::DEVICE_CPU),
    YggdrasilDecisionForestsSetLoggingLevel);

#ifdef TFDF_STOP_TRAINING_ON_INTERRUPT
namespace interruption {

tf::Status EnableUserInterruption() {
  // Detect interrupt signals.
  const bool set_signal_handler = active_learners.fetch_add(1) == 0;
  if (set_signal_handler) {
    stop_training = false;
    previous_signal_handler = std::signal(SIGINT, StopTrainingSignalHandler);
    if (previous_signal_handler == SIG_ERR) {
      TF_RETURN_IF_ERROR(tf::Status(
          static_cast<tf::errors::Code>(absl::StatusCode::kInvalidArgument),
          "Cannot change the std::signal handler."));
    }
  }
  return tf::OkStatus();
}

tf::Status DisableUserInterruption() {
  const bool restore_signal_handler = active_learners.fetch_sub(1) == 1;
  if (restore_signal_handler) {
    // Restore the previous signal handler.
    if (std::signal(SIGINT, previous_signal_handler) == SIG_ERR) {
      TF_RETURN_IF_ERROR(tf::Status(
          static_cast<tf::errors::Code>(absl::StatusCode::kInvalidArgument),
          "Cannot restore the std::signal handler."));
    }
  }
  return tf::OkStatus();
}

}  // namespace interruption
#endif

}  // namespace ops
}  // namespace tensorflow_decision_forests
