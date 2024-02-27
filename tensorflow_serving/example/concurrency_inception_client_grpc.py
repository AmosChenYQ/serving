import threading
import time
import grpc

import tensorflow as tf
from tensorflow.core.framework import tensor_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

global_start = time.time()

def save_tensor_proto(tensor_proto, file_name):
  with open(file_name, "wb") as file:
    file.write(tensor_proto.SerializeToString())


def restore_tensor_proto(file_name):
  with open(file_name, "rb") as file:
    tensor_proto = tensor_pb2.TensorProto()
    tensor_proto.ParseFromString(file.read())
    return tensor_proto

class _ResultCounter(object):
  """Counter for the prediction results."""

  def __init__(self, num_tests, concurrency):
    self._num_tests = num_tests
    self._concurrency = concurrency
    self._done = 0
    self._active = 0
    self._condition = threading.Condition()

  def inc_done(self):
    with self._condition:
      self._done += 1
      self._condition.notify()

  def dec_active(self):
    with self._condition:
      self._active -= 1
      self._condition.notify()

  def throttle(self):
    with self._condition:
      while self._active == self._concurrency:
        self._condition.wait()
      self._active += 1


def _create_rpc_callback(result_counter, start_time, client_index, cost_dict):
  """Creates RPC callback function.

  Args:
    start_time: The beginning of request starts.
    result_counter: Counter for the prediction result.
    client_index: Index of the client.
  Returns:
    The callback function.
  """
  def _callback(result_future):
    """Callback function.

    Calculates the cost time for each concurrent client

    Args:
      result_future: Result future of the RPC.
    """
    end_time = time.time()
    result_counter.dec_active()
    cost_dict[client_index].append((1000 * (end_time - global_start), 1000 * (start_time - global_start)))
    

  return _callback


def do_inference(hostport, test_data_dir, batch_size, concurrency, num_tests):
  """Tests PredictionService with concurrent requests.

  Args:
    hostport: Host:port address of the PredictionService.
    test_data_dir: The full path of tensor proto data.
    batch_size: The batch size client uses.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test to use.

  Returns:
    The classification error rate.

  Raises:
    IOError: An error occurred processing test data set.
  """
  tensor_proto_data = tf.contrib.util.make_tensor_proto(
                        restore_tensor_proto(file_name="{0}/batch_size_{1}.pb".format(test_data_dir, batch_size)), 
                        shape=[batch_size, 299, 299, 3])
  
  result_dict = dict(zip(range(concurrency), [[] for i in range(concurrency)]))
  
  channel = grpc.insecure_channel(hostport)
  result_counter = _ResultCounter(num_tests, concurrency)
  for i in range(num_tests):
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'inception'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['input'].CopyFrom(tensor_proto_data)
    result_counter.throttle()
    start_time = time.time()
    result_future = stub.Predict.future(request, 5.0)  # 5 seconds
    result_future.add_done_callback(
        _create_rpc_callback(result_counter, start_time, i % concurrency, result_dict))
  
  for i in range(concurrency):
    print(result_dict[i])


tf.app.flags.DEFINE_integer('concurrency', 5,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests', 1000, 'Number of test images')
tf.app.flags.DEFINE_integer('batch_size', 1, 'Batch size of client uses')
tf.app.flags.DEFINE_string('server', '127.0.0.1:101', 'PredictionService host:port')
tf.app.flags.DEFINE_string('test_data_dir', '/root/scripts/saved-models-from-zoo/tensor_proto_data/inception', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS

# do_inference(FLAGS.server, FLAGS.test_data_dir, FLAGS.batch_size, FLAGS.concurrency, FLAGS.num_tests)

from threading import Thread

threads = []
threads_result_dict = dict(zip(range(FLAGS.concurrency), [[] for i in range(FLAGS.concurrency)]))


def serialized_inference(hostport, test_data_dir, num_tests, batch_size, client_index):
  tensor_proto_data = tf.contrib.util.make_tensor_proto(
                        restore_tensor_proto(file_name="{0}/batch_size_{1}.pb".format(test_data_dir, batch_size)), 
                        shape=[batch_size, 299, 299, 3])
  channel = grpc.insecure_channel(hostport)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  time_cost = []
  
  for i in range(num_tests):
    start_time = time.time()
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'inception'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['input'].CopyFrom(tensor_proto_data)
    result = stub.Predict(request, 5.0)  # 5 secs timeout
    # print(result.model_spec.name)
    end_time = time.time()
    time_cost.append(1000 * (end_time - start_time))
    # time_cost.append((1000 * (start_time - global_start), 1000 * (end_time - global_start)))
  
  return client_index, time_cost


# for i in range(FLAGS.concurrency):
#   thread = Thread(target=serialized_inference, args=(FLAGS.server, FLAGS.test_data_dir, FLAGS.num_tests, FLAGS.batch_size, i))
#   threads.append(thread)

# for thread in threads:
#   thread.start()
# for thread in threads:
#   thread.join()

# global_end = time.time()
# print(global_end - global_start)
    
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
from functools import partial

# We can use a with statement to ensure threads are cleaned up promptly
'''
with concurrent.futures.ThreadPoolExecutor(max_workers=FLAGS.concurrency) as executor:
    # Start the load operations and mark each future with its URL
    future_to_time_cost = [executor.submit(serialized_inference, FLAGS.server, FLAGS.test_data_dir,  FLAGS.num_tests, FLAGS.batch_size, i) for i in range(FLAGS.concurrency)]

    print(future_to_time_cost)

    for future in concurrent.futures.as_completed(future_to_time_cost):
        try:
            id, time_cost = future.result()
            # print(time_cost)
        except Exception as exc:
            print('generated an exception: %s', exc)
        else:
            print(id, time_cost)
'''

# partial_inference = partial(serialized_inference, FLAGS.server, FLAGS.test_data_dir,  FLAGS.num_tests, FLAGS.batch_size)


time.sleep(1)

with ProcessPoolExecutor(max_workers=FLAGS.concurrency) as executor:
    # results = executor.map(partial_inference, [i for i in range(FLAGS.concurrency)])
    futures = [executor.submit(
                serialized_inference, FLAGS.server, FLAGS.test_data_dir,  FLAGS.num_tests, FLAGS.batch_size, i) for i in range(int(FLAGS.concurrency))]
    wait(futures)



for future in futures:
  print(future.result())
