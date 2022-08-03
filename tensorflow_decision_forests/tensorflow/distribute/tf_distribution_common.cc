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

#include "tensorflow_decision_forests/tensorflow/distribute/tf_distribution_common.h"

#include "absl/strings/match.h"

// TODO: Use payload on next TF release (introduced on 2021-11-02)
// #include "tensorflow/core/distributed_runtime/error_payloads.h"

namespace yggdrasil_decision_forests {
namespace distribute {

bool IsPermanentWorkerError(const absl::Status& status) {
  // TODO: Use payload on next TF release (introduced on 2021-11-02)
  //  if (status.GetPayload(tensorflow::kWorkerPossiblyRestarted).has_value()) {
  //    // Example: "TensorFlow: ABORTED: Session 87b71b3d46e250c7 is not
  //    found". return false;
  //  }

  if (absl::StrContains(status.message(), "is not found") ||
      absl::StrContains(status.message(), "NOT_FOUND")
  ) {
    return false;
  }

  return true;
}

}  // namespace distribute
}  // namespace yggdrasil_decision_forests
