##############################
# Converting to tfrecord has two ways:
# 1. tf.data
# 2. tf.python_io
# In this case, first way is chosen
##############################


import tensorflow as tf

import numpy as np


# Binarization
# 2**30 is enough for 10**9
def to_bin(x, bins=30):
    str = bin(x)[2:].zfill(bins)
    return [int(b) for b in str]


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def converter(source, target):

    dataset = tf.data.experimental.CsvDataset([source], record_defaults=[tf.int64]*31)

    # map can use tf.py_func to apply arbitrary python logic

    def _parse(*x):
        return x[0:-1],tf.one_hot(x[-1],depth=2,dtype=tf.float32)

    dataset = dataset.map(_parse)

    # for a,b in dataset.take(1):
    #     print(a)
    #     print(b)


    # tensorflow tfrecord only support string type, and must be a scalar. So need to serialized our dataset.
    # fundamentally a tf.Example is a {"string": tf.train.Feature} mapping.

    def serialize_example(x, y):
        feature = {
            'feature': _int64_feature(x),
            'label': _float_feature(y),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def tf_serialize(x,y):
        tf_string = tf.py_func(
            serialize_example,
            (x,y),
            tf.string)
        return tf.reshape(tf_string, ())

    serialized_dataset=dataset.map(tf_serialize)

    for a in serialized_dataset.take(1):
        print(a)

    writer = tf.data.experimental.TFRecordWriter(target)
    writer.write(serialized_dataset)
    print('writing finished')


if __name__ == '__main__':
    tf.enable_eager_execution()
    converter('data/dataset_1b.csv', 'data/tfrecord_1b')
