import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
import csv

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

if __name__ == '__main__':
    tf.enable_eager_execution()
    print(get_available_gpus())
    print(tf.__version__)

    source = ['tfrecord_1m']
    raw_dataset = tf.data.TFRecordDataset(source)

    for raw_record in raw_dataset.take(10):
        print(repr(raw_record))

    feature_description = {
        'feature': tf.FixedLenFeature((30),tf.int64),
        'label': tf.FixedLenFeature((2), tf.float32),
    }

    from tensorflow.python.keras.utils import to_categorical
    def _parse_function(example_proto):
        parsed_feature= tf.parse_single_example(example_proto, feature_description)
        return parsed_feature['feature'],parsed_feature['label']

    parsed_dataset = raw_dataset.map(_parse_function)

    for raw_record in parsed_dataset.take(10):
        print(repr(raw_record))