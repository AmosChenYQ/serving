from google.protobuf import text_format
from tensorflow_serving.servables.tensorflow import session_bundle_config_pb2


# Below code is trying to get text format of batching parameters
batching_parameters = session_bundle_config_pb2.BatchingParameters()
batching_parameters.max_batch_size.value = 64
batching_parameters.batch_timeout_micros.value = 20000
batching_parameters.max_enqueued_batches.value = 256
batching_parameters.num_batch_threads.value = 40
batching_parameters.allowed_batch_sizes.append(1)
batching_parameters.allowed_batch_sizes.append(2)
batching_parameters.allowed_batch_sizes.append(3)

print(text_format.MessageToString(batching_parameters))