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


#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("YggdrasilDistributeRunTask")
    .SetIsStateful()
    .Attr("welcome_blob: string")
    .Attr("worker_name: string")
    .Attr("resource_uid: string")
    .Attr("worker_idx: int")
    .Attr("parallel_execution_per_worker: int")
    .Attr("worker_addresses: list(string)")
    .Attr("worker_resource_ids: list(string)")
    .Input("input_blob: string")
    .Output("output_blob: string");

REGISTER_OP("YggdrasilDistributeRunInterWorkerTask")
    .SetIsStateful()
    .Attr("resource_uid: string")
    .Input("input_blob: string")
    .Output("output_blob: string");

REGISTER_OP("YggdrasilDistributeStopWorker")
    .SetIsStateful()
    .Input("kill_worker_manager: bool")
    .Attr("resource_uid: string");

}  // namespace tensorflow
