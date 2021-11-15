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

#ifndef THIRD_PARTY_TENSORFLOW_DECISION_FORESTS_TENSORFLOW_DISTRIBUTE_TF_DISTRIBUTION_COMMON_H_
#define THIRD_PARTY_TENSORFLOW_DECISION_FORESTS_TENSORFLOW_DISTRIBUTE_TF_DISTRIBUTION_COMMON_H_

#include "absl/status/status.h"

namespace yggdrasil_decision_forests {
namespace distribute {

// Checks whether an error returned by a worker is final or transitory (e.g.
// the worker is being resheduled).
bool IsPermanentWorkerError(const absl::Status& status);

}  // namespace distribute
}  // namespace yggdrasil_decision_forests

#endif  // THIRD_PARTY_TENSORFLOW_DECISION_FORESTS_TENSORFLOW_DISTRIBUTE_TF_DISTRIBUTION_COMMON_H_
