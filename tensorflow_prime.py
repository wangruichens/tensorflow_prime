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
    # tf.enable_eager_execution()
    print(get_available_gpus())
    print(tf.__version__)

    source = ['data/tfrecord_1b']
    raw_dataset = tf.data.TFRecordDataset(source)

    feature_description = {
        'feature': tf.FixedLenFeature((30), tf.int64),
        'label': tf.FixedLenFeature((2), tf.int64),
    }


    def _parse_function(example_proto):
        parsed_feature = tf.parse_single_example(example_proto, feature_description)
        # MatMul expected float tensor. Not accept int64
        return tf.cast(parsed_feature['feature'], tf.float32), tf.cast(parsed_feature['label'], tf.float32)


    dataset = raw_dataset.map(_parse_function,num_parallel_calls=4)
    # for r in dataset.take(1):
    #     print(r[0].shape)
    #     print(r[1].shape)
    #     print(r)

    with tf.device('/cpu:0'):
        # Functional Model
        inputs = tf.keras.layers.Input(shape=(30,))
        x = tf.keras.layers.Embedding(30, 30)(inputs)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32, return_sequences=True, kernel_initializer='Orthogonal'))(x)
        x = tf.keras.layers.GlobalMaxPool1D()(x)
        # x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dense(8, activation='relu')(x)
        predictions = tf.keras.layers.Dense(2, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=predictions)

        # Sequential Model
        # model = tf.keras.Sequential()
        # model.add(tf.keras.layers.Dense(20, activation='relu',input_dim=30))
        # model.add(tf.keras.layers.Dense(15, activation='relu'))
        # model.add(tf.keras.layers.Dense(5, activation='relu'))
        # model.add(tf.keras.layers.Dropout(0.5))
        # model.add(tf.keras.layers.Dense(2, activation='softmax'))

    # we have about 10**8 cases.
    dataset = dataset.shuffle(buffer_size=2048)
    val_test = dataset.take(1000000)
    val = val_test.take(500000).shuffle(buffer_size=2048).prefetch(2048).batch(2048).repeat()
    test = val_test.skip(500000).shuffle(buffer_size=2048).prefetch(2048).batch(2048).repeat()
    train = dataset.skip(100000).shuffle(buffer_size=2048).prefetch(2048).batch(2048).repeat()

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # tf.keras.backend.set_session(sess)


    # Multi gpu
    # model = tf.keras.utils.multi_gpu_model(model, gpus=2, cpu_merge=False)
    tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    model.fit(train, epochs=100, steps_per_epoch=100, callbacks=[tensorboard], validation_data=val,
              validation_steps=50)

    acc = model.evaluate(test, steps=200)
    print('testing acc:', acc[1])
