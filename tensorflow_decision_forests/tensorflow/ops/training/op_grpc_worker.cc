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
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

// Creates a YDF GRPC Worker server. If a server with the same key already
// exists, this is a no-op. Returns the port of the server.
//
// Args:
//     "key": Key of the server. Only one server with a given key can exist in a
//     session.
//     "force_ydf_port": Port for YDF to use. The chief and the workers should
//     be able to communicate thought this port. If -1, an available port
//     is automatically selected.
//
// Output:
//   port: Port of the GRPC server. If force_ydf_port is set, returns
//   "force_ydf_port".
REGISTER_OP("SimpleMLCreateYDFGRPCWorker")
    .SetIsStateful()
    .Attr("key: int")
    .Attr("force_ydf_port: int = -1")
    .Output("port: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return OkStatus();
    });

// Indicates to the GRPC manager that the address of a worker have changed.
// This function takes change to update the other workers for inter worker
// communication.
//
// Args:
//     "key": Key of the server.
//     "worker_idx": Index of the worker to modify.
//     "new_address": New address of the worker.
//
// Output:
//   port: Port of the GRPC server.
REGISTER_OP("SimpleMLUpdateGRPCWorkerAddress")
    .SetIsStateful()
    .Attr("key: int")
    .Input("worker_idx: int32")
    .Input("new_address: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_input_handle_shapes_and_types(
          0, {shape_inference::ShapeAndType(c->Scalar(), DataType::DT_INT32)});
      c->set_input_handle_shapes_and_types(
          1, {shape_inference::ShapeAndType(c->Scalar(), DataType::DT_STRING)});
      return OkStatus();
    });

// Stop any running YDF GRPC Worker server.
//
// Args:
//     "key": Key of the server.
//
REGISTER_OP("SimpleMLStopYDFGRPCWorker")
    .SetIsStateful()
    .Attr("key: int")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      return OkStatus();
    });

}  // namespace tensorflow
