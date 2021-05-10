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

#ifndef TENSORFLOW_DECISION_FORESTS_TENSORFLOW_OPS_TRAINING_TRAINING_H_
#define TENSORFLOW_DECISION_FORESTS_TENSORFLOW_OPS_TRAINING_TRAINING_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/utils/tensorflow.h"

namespace tensorflow_decision_forests {
namespace ops {

// A non-compiled Yggdrasil model.
class YggdrasilModelContainer : public tensorflow::ResourceBase {
 public:
  ~YggdrasilModelContainer() override = default;

  std::string DebugString() const override { return "YggdrasilModelContainer"; }

  tensorflow::Status LoadModel(const absl::string_view model_path);

  tensorflow::int64 MemoryUsed() const override {
    return approximate_model_size_in_memory_;
  }

  std::unique_ptr<yggdrasil_decision_forests::model::AbstractModel>*
  mutable_model() {
    return &model_;
  }

  const yggdrasil_decision_forests::model::AbstractModel& model() {
    return *model_;
  }

  const int num_label_classes() const { return num_label_classes_; }

  const std::vector<std::string>& output_class_representation() const {
    return output_class_representation_;
  }

 private:
  // The model.
  std::unique_ptr<yggdrasil_decision_forests::model::AbstractModel> model_;

  // Number of output classes. This information is contained in the model, but
  // cached for fast access.
  int num_label_classes_ = -1;

  // String representation of the output classes.
  std::vector<std::string> output_class_representation_;

  // Approximation of the model size in memory.
  int64_t approximate_model_size_in_memory_ = 0;
};

}  // namespace ops
}  // namespace tensorflow_decision_forests

#endif  // THIRD_PARTY_TENSORFLOW_DECISION_FORESTS_TENSORFLOW_OPS_TRAINING_H_
