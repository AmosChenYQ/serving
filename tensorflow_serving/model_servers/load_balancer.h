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

#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_LOAD_BALANCER_SERVICE_IMPL_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_LOAD_BALANCER_SERVICE_IMPL_H_

#include "tensorflow_serving/apis/load_balancer_service.grpc.pb.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

#include <memory>
#include <string>
#include <vector>


namespace tensorflow {
namespace serving {

class LoadBalancerServiceImpl final : public LoadBalancerService::Service {
 public:

  explicit LoadBalancerServiceImpl(const std::vector<std::string>& worker_list);

  ::grpc::Status Predict(::grpc::ServerContext* context,
                         const LoadBalancerRequest* request,
                         PredictResponse* response) override;

 private:
  
  std::vector<std::string> worker_list_;
  std::vector<std::unique_ptr<PredictionService::Stub>> stub_list_;

};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_LOAD_BALANCER_SERVICE_IMPL_H_
