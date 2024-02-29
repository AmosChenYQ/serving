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
"""Send JPEG image to tensorflow_model_server loaded with ResNet model.

"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.
import numpy as np
from PIL import Image
import grpc
import requests
import time

import tensorflow as tf
from tensorflow.core.framework import tensor_pb2

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

# The image URL is the location of the image we should send to the server
# IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'

tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
tf.app.flags.DEFINE_string('folder', '', 'path to image tensor proto')
FLAGS = tf.app.flags.FLAGS


def read_tensor_from_image_file_vgg19(file_name,
                                      input_height=224,
                                      input_width=224,
                                      input_mean=[123.68, 116.779, 103.939],
                                      input_std=1.0,
                                      batch_size=1):
  input_name = "file_reader"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.stack([float_caster for _ in range(batch_size)], axis=0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, input_mean), input_std)
  sess = tf.compat.v1.Session()
  result = sess.run(normalized)
  tf.reset_default_graph()
  return result


def save_tensor_proto(tensor_proto, file_name):
  with open(file_name, "wb") as file:
    file.write(tensor_proto.SerializeToString())


def restore_tensor_proto(file_name):
  with open(file_name, "rb") as file:
    tensor_proto = tensor_pb2.TensorProto()
    tensor_proto.ParseFromString(file.read())
    return tensor_proto


def generate_batched_tensor_proto():
  for batch_size in range(16):
    resized_input = read_tensor_from_image_file_vgg19(file_name=FLAGS.image, batch_size=batch_size + 1)
    tensor_proto = tf.contrib.util.make_tensor_proto(resized_input, shape=[batch_size + 1, 224, 224, 3])
    save_tensor_proto(tensor_proto=tensor_proto,
                      file_name="/root/scripts/saved-models-from-zoo/tensor_proto_data/vgg/batch_size_{0}.pb".format(batch_size+1))


def main(_):

  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  # Send request
  # See prediction_service.proto for gRPC request/response details.

  time_cost = []

  for batch_size in [1, 2, 4, 8]:
    for i in range(4):
      request = predict_pb2.PredictRequest()
      request.model_spec.name = 'default'
      request.model_spec.signature_name = 'serving_default'
      request.inputs['inputs'].CopyFrom(
          tf.contrib.util.make_tensor_proto(restore_tensor_proto(file_name="/root/scripts/saved-models-from-zoo/tensor_proto_data/vgg/batch_size_{0}.pb".format(batch_size+1)),
                                            shape=[batch_size+1, 224, 224, 3]))
      start_time = time.time()
      result = stub.Predict(request, 30.0)  # 10 secs timeout
      end_time = time.time()
      time_cost.append(1000 * (end_time - start_time))


  print(time_cost)


if __name__ == '__main__':
  tf.app.run()
