/* Copyright 2018 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_serving/model_servers/load_balancer.h"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_format.h"

#include "grpcpp/create_channel.h"
// #include "grpc/grpc.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/support/status.h"

// #include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"




ABSL_FLAG(uint16_t, port, 111, "Server port for the service");
ABSL_FLAG(std::vector<std::string>, worker_list,
          std::vector<std::string>({"127.0.0.1:101"}),
          "Workers ip address in format of host:port");

namespace tensorflow {
namespace serving {


using grpc::Channel;
using grpc::ClientContext;
// using grpc::Status;

LoadBalancerServiceImpl::LoadBalancerServiceImpl(const std::vector<std::string>& worker_list) {
  worker_list_ = worker_list;
  for (const auto& worker_addr : worker_list_) {
    std::shared_ptr<Channel> worker_channel = ::grpc::CreateChannel(worker_addr, ::grpc::InsecureChannelCredentials());
    std::unique_ptr<PredictionService::Stub> worker_stub = PredictionService::NewStub(worker_channel);
    stub_list_.push_back(std::move(worker_stub));
  }
}

::grpc::Status LoadBalancerServiceImpl::Predict(::grpc::ServerContext *context,
                                              const LoadBalancerRequest *request,
                                              PredictResponse *response) {

  const google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& req_inputs 
    = request->predict_request().inputs();
  if (req_inputs.at("input").tensor_shape().dim(0).size() != 1) {
    return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT, "Batch size should only be 1.");
  }

  PredictRequest predict_request{request->predict_request()};
  PredictResponse predict_response;
  ClientContext client_context;

  if (stub_list_.size() >= 1) {
    ::grpc::Status status = stub_list_[0]->Predict(&client_context, predict_request, response);
    return status;
  } else {
    return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT, "None rpc stub is able to be called.");
  }


  return ::grpc::Status::OK;
}

}  // namespace serving
}  // namespace tensorflow

void RunServer(uint16_t port, const std::vector<std::string>& worker_list) {
  using tensorflow::serving::LoadBalancerServiceImpl;


  std::string server_address = absl::StrFormat("0.0.0.0:%d", port);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

  LoadBalancerServiceImpl service{worker_list};
  builder.RegisterService(&service);

  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  RunServer(absl::GetFlag(FLAGS_port), absl::GetFlag(FLAGS_worker_list));
  return 0;
}
