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

// Op storing the input feature and label value in memory before the training.
//
#ifndef TENSORFLOW_DECISION_FORESTS_TENSORFLOW_OPS_TRAINING_FEATURES_H_
#define TENSORFLOW_DECISION_FORESTS_TENSORFLOW_OPS_TRAINING_FEATURES_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/utils/tensorflow.h"

namespace tensorflow_decision_forests {
namespace ops {

// Name of the tensorflow resource container i.e. the "namespace" containing
// all the resources (feature, model) created during the training.

constexpr char kModelContainer[] = "decision_forests";

// The function of a dataset.
enum class DatasetType { kTraining, kValidation };

// Container of feature values.
class AbstractFeatureResource : public ::tensorflow::ResourceBase {
 public:
  explicit AbstractFeatureResource(std::string feature_name)
      : feature_name_(feature_name) {}

  ~AbstractFeatureResource() override {}

  const std::string& feature_name() const { return feature_name_; }

  std::string DebugString() const override {
    return absl::StrCat("FeatureResource : ", feature_name_);
  }

 private:
  std::string feature_name_;
};

template <typename T>
T Identity(const T& v) {
  return v;
}

// Utility class to instantiate "AbstractFeatureResource" (i.e. feature values
// stored in resources) with simple types.
//
// Template arguments:
//   T: The storage representation (the C++ type used by the storage container).
//   V: The input representation (the TensorFlow dtype)
//   f: A conversion between the input and storage representation.
template <typename T, typename V = T, T (*f)(const V&) = Identity>
class FeatureResource : public AbstractFeatureResource {
 public:
  explicit FeatureResource(std::string feature_name)
      : AbstractFeatureResource(feature_name) {}

  // Appends the tensor values for the object cache.
  void Add(const tensorflow::Tensor& tensor) {
    tensorflow::mutex_lock l(mu_);
    num_batches_++;
    const auto tensor_data = tensor.flat<V>();
    for (int idx = 0; idx < tensor_data.size(); idx++) {
      data_.push_back(f(tensor_data(idx)));
    }
  }

  std::vector<T>* mutable_data() { return &data_; }

  const std::vector<T>& data() const { return data_; }

  tensorflow::mutex* mutable_mutex() { return &mu_; }

  int64_t NumBatches() const { return num_batches_; }

 private:
  tensorflow::mutex mu_;
  std::vector<T> data_ GUARDED_BY(mu_);
  // Number of batches of data.
  int64_t num_batches_ = 0;
};

template <>
class FeatureResource<std::string> : public AbstractFeatureResource {
 public:
  explicit FeatureResource(std::string feature_name)
      : AbstractFeatureResource(feature_name) {}

  // Appends the tensor values for the object cache.
  void Add(const tensorflow::Tensor& tensor) LOCKS_EXCLUDED(mu_) {
    tensorflow::mutex_lock l(mu_);
    num_batches_++;
    const auto tensor_data = tensor.flat<tensorflow::tstring>();
    for (int idx = 0; idx < tensor_data.size(); idx++) {
      const auto value = tensor_data(idx);
      const auto indexed_value = index_.find(value);
      if (indexed_value == index_.end()) {
        // New value.
        const auto index_value = reverse_index_.size();
        reverse_index_.push_back(value);
        index_[value] = index_value;
        data_.push_back(index_value);
      } else {
        // Existing value.
        data_.push_back(indexed_value->second);
      }
    }
  }

  tensorflow::mutex* mutable_mutex() { return &mu_; }

  int64_t NumBatches() const { return num_batches_; }

  const std::vector<int64_t>& indexed_data() const { return data_; }

  const std::vector<std::string>& reverse_index() const {
    return reverse_index_;
  }

  // Note: This function is to be used in a lambda (thread annotations don't
  // work with lambda, and tensorflow thread library doesn't support "assert
  // locked" type annotations).
  void non_mutex_protected_clear() ABSL_NO_THREAD_SAFETY_ANALYSIS {
    data_.clear();
    index_.clear();
    reverse_index_.clear();
  }

 private:
  tensorflow::mutex mu_;
  std::vector<int64_t> data_ GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, int64_t> index_ GUARDED_BY(mu_);
  std::vector<std::string> reverse_index_ GUARDED_BY(mu_);
  // Number of batches of data.
  int64_t num_batches_ = 0;
};

// Aggregates tensor values into a resource.
template <typename T, typename TResource = FeatureResource<T>,
          int Tnum_inputs = 1>
class Feature : public tensorflow::OpKernel {
 public:
  using Resource = TResource;
  static constexpr int kNumInputs = Tnum_inputs;

  explicit Feature(tensorflow::OpKernelConstruction* ctx)
      : tensorflow::OpKernel(ctx), resource_(nullptr) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("id", &identifier_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_name", &feature_name_));
  }

  ~Feature() override {
    if (resource_) {
      resource_->Unref();
      resource_ = nullptr;
    }
  }

  const std::string& feature_name() const { return feature_name_; }

  void Compute(tensorflow::OpKernelContext* ctx) override {
    tensorflow::mutex_lock l(mu_);

    if (!resource_) {
      // Note: Using tf.Estimator can create multiple duplicates of the same
      // KernelOp.
      // Note: The resource manager is indexing according to the typeid of the
      // template.
      AbstractFeatureResource* tmp_abstract_resource;
      OP_REQUIRES_OK(
          ctx, ctx->resource_manager()
                   ->LookupOrCreate<AbstractFeatureResource, true>(
                       kModelContainer, identifier_, &tmp_abstract_resource,
                       [&](AbstractFeatureResource** resource)
                           -> tensorflow::Status {
                         *resource = new Resource(feature_name_);
                         return tensorflow::Status();
                       }));
      resource_ = static_cast<Resource*>(tmp_abstract_resource);
    }
    if constexpr (kNumInputs == 1) {
      OP_REQUIRES(ctx, ctx->input(0).dims() == 1,
                  tensorflow::Status(static_cast<tsl::errors::Code>(
                                         absl::StatusCode::kInvalidArgument),
                                     "The input 0 feature should have rank 1"));
      resource_->Add(ctx->input(0));
    } else if constexpr (kNumInputs == 2) {
      OP_REQUIRES(ctx, ctx->input(0).dims() == 1,
                  tensorflow::Status(static_cast<tsl::errors::Code>(
                                         absl::StatusCode::kInvalidArgument),
                                     "The input 0 feature should have rank 1"));
      OP_REQUIRES(ctx, ctx->input(1).dims() == 1,
                  tensorflow::Status(static_cast<tsl::errors::Code>(
                                         absl::StatusCode::kInvalidArgument),
                                     "The input 1 feature should have rank 1"));
      resource_->Add(ctx->input(0), ctx->input(1));
    } else {
      YDF_LOG(FATAL) << "Invalid dimensions";
    }
  }

 private:
  tensorflow::mutex mu_;
  std::string identifier_;
  std::string feature_name_;
  Resource* resource_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(Feature);
};

// Utility class to instantiate "AbstractFeatureResource" (i.e. feature values
// stored in resources) with feature values composed of a list of simple types,
// and feed with ragged tensors.
//
// Template arguments:
//   T: The simple type storage representation (the C++ type used by the storage
//   container). V: The simple type input representation (the TensorFlow dtype)
//   f: A conversion between the input and storage representation.
//
template <typename T, typename V = T, T (*f)(const V&) = Identity>
class MultiValueRaggedFeatureResource : public AbstractFeatureResource {
 public:
  explicit MultiValueRaggedFeatureResource(std::string feature_name)
      : AbstractFeatureResource(feature_name) {
    row_splits_.push_back(0);
  }

  // Appends the tensor values for the object cache.
  void Add(const tensorflow::Tensor& values,
           const tensorflow::Tensor& row_splits) {
    tensorflow::mutex_lock l(mu_);
    const auto values_data = values.flat<V>();
    const auto row_splits_data = row_splits.flat<int64_t>();

    const auto row_splits_offset = values_.size();
    for (int idx = 0; idx < values_data.size(); idx++) {
      values_.push_back(f(values_data(idx)));
    }

    // The first element is "0" and already in "row_splits_".
    for (int idx = 1; idx < row_splits_data.size(); idx++) {
      row_splits_.push_back(row_splits_data(idx) + row_splits_offset);
    }

    num_batches_++;
  }

  void non_mutex_protected_clear() ABSL_NO_THREAD_SAFETY_ANALYSIS {
    row_splits_.assign({0});
    values_.clear();
  }

  const std::vector<T>& values() const { return values_; }

  const std::vector<size_t>& row_splits() const { return row_splits_; }

  tensorflow::mutex* mutable_mutex() { return &mu_; }

  int64_t num_examples() const { return row_splits_.size() - 1; }

  int64_t num_batches() const { return num_batches_; }

 private:
  tensorflow::mutex mu_;

  // All the following fields should be protected by "mu_".

  // The values of the i-th example are {values_[j] for j in
  // row_splits_[i]..row_splits_[i+1]-1}).
  std::vector<T> values_ GUARDED_BY(mu_);
  std::vector<size_t> row_splits_;
  // Number of batches of data.
  int64_t num_batches_ = 0;
};

class SimpleMLNumericalFeature : public Feature<float> {
 public:
  explicit SimpleMLNumericalFeature(tensorflow::OpKernelConstruction* ctx)
      : Feature(ctx) {}
};

class SimpleMLCategoricalStringFeature : public Feature<std::string> {
 public:
  explicit SimpleMLCategoricalStringFeature(
      tensorflow::OpKernelConstruction* ctx)
      : Feature(ctx) {}
};

class SimpleMLCategoricalIntFeature : public Feature<int32_t> {
 public:
  explicit SimpleMLCategoricalIntFeature(tensorflow::OpKernelConstruction* ctx)
      : Feature(ctx) {}
};

class SimpleMLCategoricalSetStringFeature
    : public Feature<std::string,
                     MultiValueRaggedFeatureResource<tensorflow::tstring>, 2> {
 public:
  explicit SimpleMLCategoricalSetStringFeature(
      tensorflow::OpKernelConstruction* ctx)
      : Feature(ctx) {}
};

class SimpleMLCategoricalSetIntFeature
    : public Feature<int32_t, MultiValueRaggedFeatureResource<int32_t>, 2> {
 public:
  explicit SimpleMLCategoricalSetIntFeature(
      tensorflow::OpKernelConstruction* ctx)
      : Feature(ctx) {}
};

inline uint64_t hasher(const tensorflow::tstring& v) {
  return yggdrasil_decision_forests::dataset::HashColumnString(v);
}

class SimpleMLHashFeature
    : public Feature<uint64_t,
                     FeatureResource<uint64_t, tensorflow::tstring, hasher>> {
 public:
  explicit SimpleMLHashFeature(tensorflow::OpKernelConstruction* ctx)
      : Feature(ctx) {}
};

// Index of "feature aggregators" (i.e. structures containing attribute values)
// and use them to create/update a VerticalDataset. The "FeatureSet" does not
// own the feature aggregators.
class FeatureSet {
 public:
  ~FeatureSet();

  // Id of the aggregator containing the label values.
  const std::string& label_feature() const;

  // Id of the aggregator containing the weight values.
  const std::string& weight_feature() const;

  // Ids of the aggregator containing the input features.
  const std::vector<std::string>& input_features() const;

  // Connects the "FeatureSet" to already existing feature aggregators.
  //
  // Args:
  //  ctx: TensorFlow context.
  //  concat_feature_ids: Comma separated list of feature ids (id for the
  //    feature accumulator) to use as input of the model.
  //  label_id: Optional id to the label feature accumulator.
  //  weight_id: Optional id to the weight feature accumulator.
  //  existing_dataspec: Optional existing dataspec to use as guide for the
  //    dataspec feature idx.
  //  dataset_type: The function of the dataset.
  tensorflow::Status Link(
      tensorflow::OpKernelContext* ctx,
      const std::vector<std::string>& column_ids,
      const ::yggdrasil_decision_forests::dataset::proto::
          DataSpecification* const existing_dataspec,
      const DatasetType dataset_type = DatasetType::kTraining);

  tensorflow::Status Unlink();

  template <typename T>
  using FeatureIterator = std::function<tensorflow::Status(
      typename T::Resource*, const int feature_idx)>;

  tensorflow::Status IterateFeatures(
      FeatureIterator<SimpleMLNumericalFeature> lambda_numerical,
      FeatureIterator<SimpleMLCategoricalStringFeature>
          lambda_categorical_string,
      FeatureIterator<SimpleMLCategoricalIntFeature> lambda_categorical_int,
      FeatureIterator<SimpleMLCategoricalSetStringFeature>
          lambda_categorical_set_string,
      FeatureIterator<SimpleMLCategoricalSetIntFeature>
          lambda_categorical_set_int,
      FeatureIterator<SimpleMLHashFeature> lambda_hash);

  // Initialize a dataset (including the dataset's dataspec) from the linked
  // resource aggregators.
  tensorflow::Status InitializeDatasetFromFeatures(
      tensorflow::OpKernelContext* ctx,
      const ::yggdrasil_decision_forests::dataset::proto::
          DataSpecificationGuide& guide,
      ::yggdrasil_decision_forests::dataset::VerticalDataset* dataset);

  // Moves the feature values contained in the aggregators into the dataset.
  // Following this call, the feature aggregators are empty.
  tensorflow::Status MoveExamplesFromFeaturesToDataset(
      tensorflow::OpKernelContext* ctx,
      ::yggdrasil_decision_forests::dataset::VerticalDataset* dataset);

  int NumFeatures() const;

 private:
  std::vector<std::pair<int, SimpleMLNumericalFeature::Resource*>>
      numerical_features_;
  std::vector<std::pair<int, SimpleMLCategoricalStringFeature::Resource*>>
      categorical_string_features_;
  std::vector<std::pair<int, SimpleMLCategoricalIntFeature::Resource*>>
      categorical_int_features_;
  std::vector<std::pair<int, SimpleMLHashFeature::Resource*>> hash_features_;
  std::vector<std::pair<int, SimpleMLCategoricalSetStringFeature::Resource*>>
      categorical_set_string_features_;
  std::vector<std::pair<int, SimpleMLCategoricalSetIntFeature::Resource*>>
      categorical_set_int_features_;
};

}  // namespace ops
}  // namespace tensorflow_decision_forests

#endif  // THIRD_PARTY_TENSORFLOW_DECISION_FORESTS_TENSORFLOW_OPS_TRAINING_H_
