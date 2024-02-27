# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow_serving/apis/load_balancer_service.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from tensorflow_serving.apis import load_balancer_pb2 as tensorflow__serving_dot_apis_dot_load__balancer__pb2
from tensorflow_serving.apis import predict_pb2 as tensorflow__serving_dot_apis_dot_predict__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow_serving/apis/load_balancer_service.proto',
  package='tensorflow.serving',
  syntax='proto3',
  serialized_options=_b('\370\001\001'),
  serialized_pb=_b('\n3tensorflow_serving/apis/load_balancer_service.proto\x12\x12tensorflow.serving\x1a+tensorflow_serving/apis/load_balancer.proto\x1a%tensorflow_serving/apis/predict.proto2n\n\x13LoadBalancerService\x12W\n\x07Predict\x12\'.tensorflow.serving.LoadBalancerRequest\x1a#.tensorflow.serving.PredictResponseB\x03\xf8\x01\x01\x62\x06proto3')
  ,
  dependencies=[tensorflow__serving_dot_apis_dot_load__balancer__pb2.DESCRIPTOR,tensorflow__serving_dot_apis_dot_predict__pb2.DESCRIPTOR,])



_sym_db.RegisterFileDescriptor(DESCRIPTOR)


DESCRIPTOR._options = None

_LOADBALANCERSERVICE = _descriptor.ServiceDescriptor(
  name='LoadBalancerService',
  full_name='tensorflow.serving.LoadBalancerService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=159,
  serialized_end=269,
  methods=[
  _descriptor.MethodDescriptor(
    name='Predict',
    full_name='tensorflow.serving.LoadBalancerService.Predict',
    index=0,
    containing_service=None,
    input_type=tensorflow__serving_dot_apis_dot_load__balancer__pb2._LOADBALANCERREQUEST,
    output_type=tensorflow__serving_dot_apis_dot_predict__pb2._PREDICTRESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_LOADBALANCERSERVICE)

DESCRIPTOR.services_by_name['LoadBalancerService'] = _LOADBALANCERSERVICE

# @@protoc_insertion_point(module_scope)
