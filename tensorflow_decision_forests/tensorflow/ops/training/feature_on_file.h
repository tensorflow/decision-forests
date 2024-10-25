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

// Ops to store feature values on the file-system in the partial dataset
// cache format (see yggdrasil_decision_forests/learner/
// distributed_decision_tree/dataset_cache/dataset_cache.h).
//
// Each worker is expected to instantiate a SimpleML*FeatureOnFile op for each
// feature. All the features should have the same "model_id" and "dataset_path".
// On each worker, once the data collection is done,
// "SimpleMLWorkerFinalizeFeatureOnFile" should be called. Once
// "SimpleMLWorkerFinalizeFeatureOnFile" has been called by all the workers,
// "SimpleMLChiefFinalizeFeatureOnFile" should be called once (this last
// operation does not require a lot of disk IO and can be done by the manager).
//
// Each worker saves the values for each feature in one separate file. So
// in the end the values of each feature is split on <number of workers> files
// under the provided <data_path>.
#ifndef TENSORFLOW_DECISION_FORESTS_TENSORFLOW_OPS_TRAINING_FEATURE_ON_FILE_H_
#define TENSORFLOW_DECISION_FORESTS_TENSORFLOW_OPS_TRAINING_FEATURE_ON_FILE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow_decision_forests/tensorflow/ops/training/features.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/column_cache.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace tensorflow_decision_forests {
namespace ops {

// Flag file used to detect already existing dataset (e.g. for when a worker or
// a manager is rescheduled).
constexpr char kFilenameDone[] = "partial_done";

// Create a done file in "dataset_path".
absl::Status CreateDoneFile(const std::string& dataset_path);

// Checks is a done file exist in "dataset_path".
bool HasDoneFile(const std::string& dataset_path);

// Receive features values (as Tensor) and export them to file (in the partial
// dataset cache format).
//
// The exact path of the created files is controlled by the partial dataset
// cache library. The files are guaranteed to be in a subdirectory of
// "dataset_path".
//
// In the case of distributed training, one such resource is instantiated on
// each worker. Each such instance is responsible for a subset of the examples
// of a single feature.
class AbstractFeatureResourceOnFile : public ::tensorflow::ResourceBase {
 public:
  typedef ::yggdrasil_decision_forests::model::distributed_decision_tree::
      dataset_cache::proto::PartialColumnShardMetadata
          PartialColumnShardMetadata;

  AbstractFeatureResourceOnFile(int feature_idx,
                                const std::string& feature_name,
                                const std::string& dataset_path,
                                const int worker_idx)
      : feature_idx_(feature_idx),
        feature_name_(feature_name),
        dataset_path_(dataset_path),
        worker_idx_(worker_idx) {}

  ~AbstractFeatureResourceOnFile() override {}

  std::string DebugString() const override {
    return "AbstractFeatureResourceOnFile";
  }

  // Prepares for the data collection.
  // Should be only called once before any value is feed.
  virtual absl::Status Begin() = 0;

  // Consumes on batch of data. Thread safe.
  absl::Status AddValue(const tensorflow::Tensor& tensor);

  // Implementation of AddValue. Thread compatible.
  virtual absl::Status AddValueImp(const tensorflow::Tensor& tensor) = 0;

  // Finalize the data collection. Should be only called once.
  absl::Status End();

  // Implementation of "End". This method is responsible to set the "type"
  // specific attribute in the "meta_data" (e.g. the fields in "type::numerical"
  // for a numerical feature).
  virtual absl::Status EndImp(PartialColumnShardMetadata* meta_data) = 0;

 protected:
  int feature_idx_;
  std::string feature_name_;
  std::string dataset_path_;
  int worker_idx_;
  tensorflow::mutex mu_;
};

class NumericalResourceOnFile : public AbstractFeatureResourceOnFile {
 public:
  NumericalResourceOnFile(int feature_idx, const std::string& feature_name,
                          const std::string& dataset_path, const int worker_idx)
      : AbstractFeatureResourceOnFile(feature_idx, feature_name, dataset_path,
                                      worker_idx) {}

  absl::Status Begin() override;
  absl::Status AddValueImp(const tensorflow::Tensor& tensor) override;
  absl::Status EndImp(PartialColumnShardMetadata* meta_data) override;

 private:
  std::unique_ptr<
      ::yggdrasil_decision_forests::model::distributed_decision_tree::
          dataset_cache::FloatColumnWriter>
      writer_;
  int64_t num_examples_ = 0;
  int64_t num_missing_examples_ = 0;
  int64_t num_batches_ = 0;
  double sum_values_ = 0;
  double min_value_ = 0;
  double max_value_ = 0;

  // At least one non-missing values observed so far.
  bool did_see_non_missing_value_ = false;
};

class CategoricalResourceOnFile : public AbstractFeatureResourceOnFile {
 public:
  CategoricalResourceOnFile(int feature_idx, const std::string& feature_name,
                            const std::string& dataset_path,
                            const int worker_idx)
      : AbstractFeatureResourceOnFile(feature_idx, feature_name, dataset_path,
                                      worker_idx) {}

  absl::Status Begin() override;
  absl::Status AddValueImp(const tensorflow::Tensor& tensor) override;
  absl::Status EndImp(PartialColumnShardMetadata* meta_data) override;

 private:
  std::unique_ptr<
      ::yggdrasil_decision_forests::model::distributed_decision_tree::
          dataset_cache::IntegerColumnWriter>
      writer_;
  int64_t num_examples_ = 0;
  int64_t num_missing_examples_ = 0;
  int32_t number_of_unique_values_ = 1;
};

class CategoricalStringResourceOnFile : public AbstractFeatureResourceOnFile {
 public:
  CategoricalStringResourceOnFile(int feature_idx,
                                  const std::string& feature_name,
                                  const std::string& dataset_path,
                                  const int worker_idx)
      : AbstractFeatureResourceOnFile(feature_idx, feature_name, dataset_path,
                                      worker_idx) {}

  absl::Status Begin() override;
  absl::Status AddValueImp(const tensorflow::Tensor& tensor) override;
  absl::Status EndImp(PartialColumnShardMetadata* meta_data) override;

 private:
  std::unique_ptr<
      ::yggdrasil_decision_forests::model::distributed_decision_tree::
          dataset_cache::IntegerColumnWriter>
      writer_;
  int64_t num_examples_ = 0;
  int64_t num_missing_examples_ = 0;

  struct Item {
    int index;
    int count;
  };
  absl::flat_hash_map<std::string, Item> items_;
};

// TF OP that aggregates tensor values into a AbstractFeatureResourceOnFile.
template <typename Resource>
class FeatureOnFileOp : public tensorflow::OpKernel {
 public:
  explicit FeatureOnFileOp(tensorflow::OpKernelConstruction* ctx)
      : tensorflow::OpKernel(ctx), resource_(nullptr) {
    static_assert(
        std::is_base_of<AbstractFeatureResourceOnFile, Resource>::value,
        "The template class argument does not derive "
        "AbstractFeatureResourceOnFile.");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_idx", &feature_idx_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_name", &feature_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dataset_path", &dataset_path_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("resource_id", &resource_id_));

    dataset_already_on_disk_ = HasDoneFile(dataset_path_);

    // TODO: Use the following code when tf2.7 is released.
    // worker_idx_ = ctx->device()->parsed_name().task;

    auto* device = dynamic_cast<tensorflow::Device*>(ctx->device());
    if (device == nullptr) {
      OP_REQUIRES_OK(ctx,
                     absl::InvalidArgumentError("Cannot find the worker idx"));
    }
    worker_idx_ = device->parsed_name().task;

    if (dataset_already_on_disk_) {
      LOG(INFO) << "Already existing dataset cache for worker #" << worker_idx_
                << " on device " << ctx->device()->name();
    }
  }

  ~FeatureOnFileOp() override {
    if (resource_) {
      resource_->Unref();
      resource_ = nullptr;
    }
  }

  const std::string& feature_name() const { return feature_name_; }

  void Compute(tensorflow::OpKernelContext* ctx) override {
    if (dataset_already_on_disk_) {
      return;
    }

    tensorflow::mutex_lock l(mu_);
    OP_REQUIRES(
        ctx, ctx->input(0).dims() == 1,
        absl::InvalidArgumentError("The input 0 feature should have rank 1"));
    if (!resource_) {
      AbstractFeatureResourceOnFile* abstract_resource;
      OP_REQUIRES_OK(
          ctx, ctx->resource_manager()
                   ->LookupOrCreate<AbstractFeatureResourceOnFile, true>(
                       kModelContainer, resource_id_, &abstract_resource,
                       [&](AbstractFeatureResourceOnFile** resource)
                           -> absl::Status {
                         *resource = new Resource(feature_idx_, feature_name_,
                                                  dataset_path_, worker_idx_);
                         return (*resource)->Begin();
                       }));
      resource_ = static_cast<Resource*>(abstract_resource);
    }
    OP_REQUIRES(ctx, ctx->input(0).dims() == 1,
                absl::InvalidArgumentError("The input should have rank 1"));
    OP_REQUIRES_OK(ctx, resource_->AddValue(ctx->input(0)));
  }

 private:
  tensorflow::mutex mu_;
  std::string resource_id_;
  int feature_idx_;
  std::string feature_name_;
  std::string dataset_path_;
  int worker_idx_;
  bool dataset_already_on_disk_;
  Resource* resource_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(FeatureOnFileOp);
};

class SimpleMLNumericalFeatureOnFile
    : public FeatureOnFileOp<NumericalResourceOnFile> {
 public:
  explicit SimpleMLNumericalFeatureOnFile(tensorflow::OpKernelConstruction* ctx)
      : FeatureOnFileOp(ctx) {}
};

class SimpleMLCategoricalIntFeatureOnFile
    : public FeatureOnFileOp<CategoricalResourceOnFile> {
 public:
  explicit SimpleMLCategoricalIntFeatureOnFile(
      tensorflow::OpKernelConstruction* ctx)
      : FeatureOnFileOp(ctx) {}
};

class SimpleMLCategoricalStringFeatureOnFile
    : public FeatureOnFileOp<CategoricalStringResourceOnFile> {
 public:
  explicit SimpleMLCategoricalStringFeatureOnFile(
      tensorflow::OpKernelConstruction* ctx)
      : FeatureOnFileOp(ctx) {}
};

// See description in op.cc.
class SimpleMLWorkerFinalizeFeatureOnFile : public tensorflow::OpKernel {
 public:
  explicit SimpleMLWorkerFinalizeFeatureOnFile(
      tensorflow::OpKernelConstruction* ctx)
      : tensorflow::OpKernel(ctx) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("feature_resource_ids", &feature_resource_ids_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dataset_path", &dataset_path_));
  }

  ~SimpleMLWorkerFinalizeFeatureOnFile() override {}

  void Compute(tensorflow::OpKernelContext* ctx) override;

 private:
  std::vector<std::string> feature_resource_ids_;
  std::string dataset_path_;

  TF_DISALLOW_COPY_AND_ASSIGN(SimpleMLWorkerFinalizeFeatureOnFile);
};

// See description in op.cc.
class SimpleMLChiefFinalizeFeatureOnFile : public tensorflow::OpKernel {
 public:
  explicit SimpleMLChiefFinalizeFeatureOnFile(
      tensorflow::OpKernelConstruction* ctx)
      : tensorflow::OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_names", &feature_names_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dataset_path", &dataset_path_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_shards", &num_shards_));
  }

  ~SimpleMLChiefFinalizeFeatureOnFile() override {}

  void Compute(tensorflow::OpKernelContext* ctx) override;

 private:
  std::vector<std::string> feature_names_;
  std::string dataset_path_;
  int num_shards_;

  TF_DISALLOW_COPY_AND_ASSIGN(SimpleMLChiefFinalizeFeatureOnFile);
};

}  // namespace ops
}  // namespace tensorflow_decision_forests

#endif  // TENSORFLOW_DECISION_FORESTS_TENSORFLOW_OPS_TRAINING_FEATURE_ON_FILE_H_
