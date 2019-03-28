##############################
# Converting to tfrecord has two ways:
# 1. tf.data
# 2. tf.python_io
# In this case, first way is chosen
##############################


import tensorflow as tf
import pickle
from tensorflow.python.keras.utils import to_categorical
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

    d=pickle.load(open(source,'rb'))

    # y=np.hsplit(d,indices_or_sections=2)

    print(d[10])
    print(d.shape)

    dataset = tf.data.experimental.CsvDataset([source], record_defaults=[tf.int64, tf.float32])

    # Using tf.py_func to applying arbitrary python logic
    # should always use tensorflow operations if possible for performance reason

    # actually this already can be passed to model.

    def _parse(x, y):
        return (to_bin(x), to_categorical(y, num_classes=2))

    dataset = dataset.map(lambda x, y: tuple(tf.py_func(_parse, [x, y], [tf.int64, tf.float32])))

    ############################ Testing ############################
    # for a,b in dataset.take(1):
    #     print(a)
    #     print(b)

    # next_element = dataset.make_one_shot_iterator().get_next()
    # with tf.Session() as sess:
    #   while True:
    #     try:
    #       print(sess.run(next_element))
    #     except tf.errors.OutOfRangeError:
    #       break
    ############################ Testing ############################

    # tensorflow tfrecord only support string type, and must be a scalar. So need to serialized our dataset.
    # using tf.Example
    # Fundamentally a tf.Example is a {"string": tf.train.Feature} mapping.

    def my_serialize(x, y):
        feature = {
            'feature': _int64_feature(x),
            'label': _float_feature(y),
        }
        # using tf.train.Example to serialize feature message
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def tf_serialize(x, y):
        tf_string = tf.py_func(
            my_serialize,
            (x, y),  # pass these args to the above function.
            tf.string)  # the return type is <a href="../../api_docs/python/tf#string"><code>tf.string</code></a>.
        return tf.reshape(tf_string, ())  # The result is a scalar

    serialized_dataset = dataset.map(tf_serialize)

    # tf.Tensor(b'\nF\n\x15\n\x05label\x12\x0c\x12\n\n\x08\x00\x00\x80?\x00\x00\x00\x00\n-\n\x07feature\x12"\x1a \n\x1e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01', shape=(), dtype=string)
    for a in serialized_dataset.take(1):
        print(a)
        print(tf.train.Example.FromString(a))

    writer = tf.data.experimental.TFRecordWriter(target)
    writer.write(serialized_dataset)
    print('writing finished')


if __name__ == '__main__':
    tf.enable_eager_execution()
    converter('dataset.pkl', 'dataset_tf')
