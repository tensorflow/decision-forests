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

#include "tensorflow_decision_forests/tensorflow/ops/training/feature_on_file.h"

#include "yggdrasil_decision_forests/utils/logging.h"

namespace tensorflow_decision_forests {
namespace ops {
namespace {

namespace tf = ::tensorflow;
namespace utils = ::yggdrasil_decision_forests::utils;
namespace dist_dt =
    ::yggdrasil_decision_forests::model::distributed_decision_tree;

using dist_dt::dataset_cache::FloatColumnWriter;
using dist_dt::dataset_cache::IntegerColumnWriter;
using dist_dt::dataset_cache::kFilenameMetaDataPostfix;
using dist_dt::dataset_cache::kFilenamePartialMetaData;
using dist_dt::dataset_cache::PartialRawColumnFileDirectory;
using dist_dt::dataset_cache::PartialRawColumnFilePath;
using dist_dt::dataset_cache::proto::PartialDatasetMetadata;

}  // namespace

tf::Status CreateDoneFile(const std::string& dataset_path) {
  return WriteStringToFile(tensorflow::Env::Default(),
                           tf::io::JoinPath(dataset_path, kFilenameDone),
                           "done");
}

bool HasDoneFile(const std::string& dataset_path) {
  return tensorflow::Env::Default()
      ->FileExists(tf::io::JoinPath(dataset_path, kFilenameDone))
      .ok();
}

absl::Status AbstractFeatureResourceOnFile::AddValue(
    const tensorflow::Tensor& tensor) {
  tensorflow::mutex_lock lock(mu_);
  return AddValueImp(tensor);
}

absl::Status AbstractFeatureResourceOnFile::End() {
  PartialColumnShardMetadata meta_data;
  RETURN_IF_ERROR(EndImp(&meta_data));

  RETURN_IF_ERROR(utils::ToUtilStatus(WriteBinaryProto(
      tensorflow::Env::Default(),
      absl::StrCat(
          PartialRawColumnFilePath(dataset_path_, feature_idx_, worker_idx_),
          kFilenameMetaDataPostfix),
      meta_data)));
  return absl::OkStatus();
}

absl::Status NumericalResourceOnFile::Begin() {
  RETURN_IF_ERROR(
      utils::ToUtilStatus(tensorflow::Env::Default()->RecursivelyCreateDir(
          PartialRawColumnFileDirectory(dataset_path_, feature_idx_))));

  writer_ = absl::make_unique<FloatColumnWriter>();
  return writer_->Open(
      PartialRawColumnFilePath(dataset_path_, feature_idx_, worker_idx_));
}

absl::Status NumericalResourceOnFile::AddValueImp(
    const tensorflow::Tensor& tensor) {
  const auto tensor_data = tensor.flat<float>();
  num_batches_++;
  num_examples_ += tensor_data.size();

  for (int value_idx = 0; value_idx < tensor_data.size(); value_idx++) {
    float value = tensor_data(value_idx);
    if (std::isnan(value)) {
      num_missing_examples_++;
    } else {
      sum_values_ += value;
      if (!did_see_non_missing_value_ || value < min_value_) {
        min_value_ = value;
      }
      if (!did_see_non_missing_value_ || value > max_value_) {
        max_value_ = value;
      }
      did_see_non_missing_value_ = true;
    }
  }

  return writer_->WriteValues(
      absl::Span<const float>(tensor_data.data(), tensor_data.size()));
}

absl::Status NumericalResourceOnFile::EndImp(
    PartialColumnShardMetadata* meta_data) {
  YDF_LOG(INFO) << "[worker] End for " << feature_name_ << ":" << feature_idx_
                << " on worker #" << worker_idx_ << " with " << num_examples_
                << " examples and " << num_batches_ << " batches";
  meta_data->set_num_examples(num_examples_);
  meta_data->set_num_missing_examples(num_missing_examples_);
  auto* numerical = meta_data->mutable_numerical();
  if (did_see_non_missing_value_) {
    numerical->set_mean(sum_values_ / (num_examples_ - num_missing_examples_));
    numerical->set_min(min_value_);
    numerical->set_max(max_value_);
  }
  return writer_->Close();
}

absl::Status CategoricalResourceOnFile::Begin() {
  RETURN_IF_ERROR(
      utils::ToUtilStatus(tensorflow::Env::Default()->RecursivelyCreateDir(
          PartialRawColumnFileDirectory(dataset_path_, feature_idx_))));

  writer_ = absl::make_unique<IntegerColumnWriter>();
  return writer_->Open(
      PartialRawColumnFilePath(dataset_path_, feature_idx_, worker_idx_),
      std::numeric_limits<int32_t>::max());
}

absl::Status CategoricalResourceOnFile::AddValueImp(
    const tensorflow::Tensor& tensor) {
  const auto tensor_data = tensor.flat<int32_t>();
  num_examples_ += tensor_data.size();
  for (int value_idx = 0; value_idx < tensor_data.size(); value_idx++) {
    int32_t value = tensor_data(value_idx);
    if (value < 0) {
      num_missing_examples_++;
    } else {
      if (value + 1 > number_of_unique_values_) {
        number_of_unique_values_ = value + 1;
      }
    }
  }

  return writer_->WriteValues(
      absl::Span<const int32_t>(tensor_data.data(), tensor_data.size()));
}

absl::Status CategoricalResourceOnFile::EndImp(
    PartialColumnShardMetadata* meta_data) {
  YDF_LOG(INFO) << "[worker] End for " << feature_name_ << ":" << feature_idx_
                << " on worker #" << worker_idx_;
  meta_data->set_num_examples(num_examples_);
  meta_data->set_num_missing_examples(num_missing_examples_);
  meta_data->mutable_categorical()->set_number_of_unique_values(
      number_of_unique_values_);
  return writer_->Close();
}

absl::Status CategoricalStringResourceOnFile::Begin() {
  RETURN_IF_ERROR(
      utils::ToUtilStatus(tensorflow::Env::Default()->RecursivelyCreateDir(
          PartialRawColumnFileDirectory(dataset_path_, feature_idx_))));

  writer_ = absl::make_unique<IntegerColumnWriter>();
  return writer_->Open(
      PartialRawColumnFilePath(dataset_path_, feature_idx_, worker_idx_),
      std::numeric_limits<int32_t>::max());
}

absl::Status CategoricalStringResourceOnFile::AddValueImp(
    const tensorflow::Tensor& tensor) {
  const auto tensor_data = tensor.flat<tensorflow::tstring>();
  std::vector<int32_t> int_values;
  int_values.reserve(tensor_data.size());

  num_examples_ += tensor_data.size();
  for (int value_idx = 0; value_idx < tensor_data.size(); value_idx++) {
    const std::string& value = tensor_data(value_idx);
    if (value.empty()) {
      num_missing_examples_++;
      int_values.push_back(-1);
    } else {
      auto value_it = items_.find(value);
      if (value_it == items_.end()) {
        const int index = static_cast<int>(items_.size());
        int_values.push_back(index);
        items_[value] = {/*index=*/index,
                         /*count=*/1};
      } else {
        value_it->second.count++;
        int_values.push_back(value_it->second.index);
      }
    }
  }

  return writer_->WriteValues(absl::Span<const int32_t>(int_values));
}

absl::Status CategoricalStringResourceOnFile::EndImp(
    PartialColumnShardMetadata* meta_data) {
  YDF_LOG(INFO) << "[worker] End for " << feature_name_ << ":" << feature_idx_
                << " on worker #" << worker_idx_;
  meta_data->set_num_examples(num_examples_);
  meta_data->set_num_missing_examples(num_missing_examples_);
  auto* categorical = meta_data->mutable_categorical();
  for (const auto& src_item : items_) {
    auto& dst_item = (*categorical->mutable_items())[src_item.first];
    dst_item.set_count(src_item.second.count);
    dst_item.set_index(src_item.second.index);
  }
  return writer_->Close();
}

void SimpleMLWorkerFinalizeFeatureOnFile::Compute(
    tensorflow::OpKernelContext* ctx) {
  YDF_LOG(INFO) << "[Feature] SimpleMLWorkerFinalizeDiskFeature on device "
                << ctx->device()->name();

  if (HasDoneFile(dataset_path_)) {
    return;
  }

  for (const auto& feature_resource_id : feature_resource_ids_) {
    AbstractFeatureResourceOnFile* abstract_resource;
    const auto lookup_status = ctx->resource_manager()->Lookup(
        kModelContainer, feature_resource_id, &abstract_resource);
    if (!lookup_status.ok()) {
      YDF_LOG(INFO) << "Resource "
                    << " not found on " << ctx->device()->name();
      return;
    }
    OP_REQUIRES_OK(ctx, utils::FromUtilStatus(abstract_resource->End()));
    abstract_resource->Unref();
  }
}

void SimpleMLChiefFinalizeFeatureOnFile::Compute(
    tensorflow::OpKernelContext* ctx) {
  YDF_LOG(INFO) << "[Feature] SimpleMLChiefFinalizeDiskFeature on device "
                << ctx->device()->name();

  if (HasDoneFile(dataset_path_)) {
    return;
  }

  YDF_LOG(INFO) << "Finalizing dataset";

  PartialDatasetMetadata meta_data;
  *meta_data.mutable_column_names() = {feature_names_.begin(),
                                       feature_names_.end()};
  meta_data.set_num_shards(num_shards_);

  OP_REQUIRES_OK(ctx,
                 WriteBinaryProto(
                     tensorflow::Env::Default(),
                     tf::io::JoinPath(dataset_path_, kFilenamePartialMetaData),
                     meta_data));

  OP_REQUIRES_OK(ctx, CreateDoneFile(dataset_path_));
}

}  // namespace ops
}  // namespace tensorflow_decision_forests
