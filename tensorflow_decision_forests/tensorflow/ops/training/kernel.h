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

// Status of long running operations.
enum LongRunningProcessStatus {
  kInProgress = 0,
  kSuccess = 1,
};

// Starts a long running process. Returns the id of the process.
absl::StatusOr<int32_t> StartLongRunningProcess(
    ::tensorflow::OpKernelContext* ctx, std::function<absl::Status()>&& call);

// Checks the status of a long running process. If the returned status of a
// process is success or failure, the status of the process should not be
// queried again.
absl::StatusOr<LongRunningProcessStatus> GetLongRunningProcessStatus(
    ::tensorflow::OpKernelContext* ctx, int32_t process_id);

// If TFDF_STOP_TRAINING_ON_INTERRUPT is set, model training is interrupted with
// the "set_stop_training_trigger()" API when the process receives a "program
// interrupt" signal (i.e. SIGINT).
//
// An interrupted model is a valid model. But its quality is likely inferior to
// a fully trained model. See "set_stop_training_trigger" for details about the
// effect of the interruption.
//
#ifdef TFDF_STOP_TRAINING_ON_INTERRUPT

// The logic in this namespace is used to interrupt the training of a model (by
// setting "stop_training_trigger=True") when receiving an interruption (e.g.
// the user pressed ctrl+c).
namespace interruption {
// Should the current training learners be stopped?
inline std::atomic<bool> stop_training;

// The interruption signal handler to restore when all the learners are done
// training.
inline void (*previous_signal_handler)(int);

// Number of learners training.
inline std::atomic<int> active_learners{0};

inline void StopTrainingSignalHandler(int signal) { stop_training = true; }

// Enables the interruption listener.
tensorflow::Status EnableUserInterruption();

// Disable the interruption listener
tensorflow::Status DisableUserInterruption();

}  // namespace interruption
#endif

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
