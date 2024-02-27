# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import grpc
import requests
import time

import tensorflow as tf
from tensorflow.core.framework import tensor_pb2
from tensorflow_serving.apis import load_balancer_pb2
from tensorflow_serving.apis import load_balancer_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2

tf.app.flags.DEFINE_string('server', '127.0.0.1:111x  ',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('folder', '', 'path to image tensor proto')
FLAGS = tf.app.flags.FLAGS


def save_tensor_proto(tensor_proto, file_name):
  with open(file_name, "wb") as file:
    file.write(tensor_proto.SerializeToString())


def restore_tensor_proto(file_name):
  with open(file_name, "rb") as file:
    tensor_proto = tensor_pb2.TensorProto()
    tensor_proto.ParseFromString(file.read())
    return tensor_proto


def main(_):
  if not FLAGS.folder:
    raise ValueError("Not data provided")

  channel = grpc.insecure_channel(FLAGS.server)
  stub = load_balancer_service_pb2_grpc.LoadBalancerServiceStub(channel)

  time_cost = []

  if FLAGS.folder:
    # Send request
    for i in range(4):
      start_time = time.time()
      load_balancer_request = load_balancer_pb2.LoadBalancerRequest()
      predict_request = predict_pb2.PredictRequest()
      predict_request.model_spec.name = 'inception'
      predict_request.model_spec.signature_name = 'serving_default'
      predict_request.inputs['input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(
          restore_tensor_proto(file_name="{0}/batch_size_{1}.pb".format(FLAGS.folder, 1)), shape=[1, 299, 299, 3]
        )
      )
      load_balancer_request.predict_request.CopyFrom(predict_request) 
      load_balancer_request.slo_target = 200

      result = stub.Predict(load_balancer_request, 10.0)  # 10 secs timeout
      print(result.outputs['output'])
      end_time = time.time()
      time_cost.append(1000 * (end_time - start_time))
  
  print(time_cost)



if __name__ == '__main__':
  tf.app.run()
