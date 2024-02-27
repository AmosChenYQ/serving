# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from tensorflow_serving.apis import load_balancer_pb2 as tensorflow__serving_dot_apis_dot_load__balancer__pb2
from tensorflow_serving.apis import predict_pb2 as tensorflow__serving_dot_apis_dot_predict__pb2


class LoadBalancerServiceStub(object):
  """open source marker; do not remove
  LoadBalancerService acts as a load balancer which work at the 
  level of application layer  
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Predict = channel.unary_unary(
        '/tensorflow.serving.LoadBalancerService/Predict',
        request_serializer=tensorflow__serving_dot_apis_dot_load__balancer__pb2.LoadBalancerRequest.SerializeToString,
        response_deserializer=tensorflow__serving_dot_apis_dot_predict__pb2.PredictResponse.FromString,
        )


class LoadBalancerServiceServicer(object):
  """open source marker; do not remove
  LoadBalancerService acts as a load balancer which work at the 
  level of application layer  
  """

  def Predict(self, request, context):
    """Predict -- Behaves like Predict method in PredictService, but it will decide
    batch size in load balancer side.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_LoadBalancerServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Predict': grpc.unary_unary_rpc_method_handler(
          servicer.Predict,
          request_deserializer=tensorflow__serving_dot_apis_dot_load__balancer__pb2.LoadBalancerRequest.FromString,
          response_serializer=tensorflow__serving_dot_apis_dot_predict__pb2.PredictResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'tensorflow.serving.LoadBalancerService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
