import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
import csv

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print(get_available_gpus())
print(tf.__version__)

def list_primes(n):
    if n < 3:
        return 0
    primes = [True] * n
    primes[0] = primes[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if primes[i]:
            primes[i * i: n: i] = [False] * len(primes[i * i: n: i])
    return primes


def to_bin(x,bins=30):
    str=bin(x)[2:].zfill(bins)
    return [int(b) for b in str]


lens=10**5
l=list_primes(lens)
pos=[]
for index,v in enumerate(l):
    if v:
        pos.append(index)

n=2
p=pos.pop(0)
data_x=[]
data_y=[]
primenum=0
while n <lens:
    if n < p:
        # if all input data are not even, then the DNN can not find any pattern
        # if np.random.uniform(0, 1) > 0.8 and n%2 !=0 :
        if np.random.uniform(0, 1) > 0.9 :
            data_x.append(to_bin(n))
            data_y.append(0.0)
    if n==p:
        primenum+=1
        data_x.append(to_bin(n))
        data_y.append(1.0)
        if len(pos)>0:
            p=pos.pop(0)
        else:
            p=lens+1

    n+=1

# Another case : determin odd or even
# n=2
# data_x=[]
# data_y=[]
# primenum=0
# while n <lens:
#     if n % 2 ==0:
#         data_x.append(to_bin(n))
#         data_y.append(0.0)
#         primenum+=1
#     else:
#         data_x.append(to_bin(n))
#         data_y.append(1.0)
#     n+=1


print('positive cases num:',primenum)
print('total cases :',len(data_x))

from sklearn.model_selection import train_test_split

data_x=np.array(data_x)
data_y=np.array(data_y)
# data_y=np.expand_dims(np.array(data_y),axis=1)

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1, random_state=None,shuffle=True)

y_train=tf.keras.utils.to_categorical(y_train,num_classes=2)
y_test=tf.keras.utils.to_categorical(y_test,num_classes=2)

print('training dataset x:' ,x_train.shape)
print('training dataset y:' ,y_train.shape)

print('testing dataset x:' ,x_test.shape)
print('testing dataset y:' ,y_test.shape)
for i in range(3):
    print(x_test[i],y_test[i])

# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

# Should not use dropout here.
with tf.device('/cpu:0'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(20,activation='relu',input_shape=(30,)))
    # model.add(tf.keras.layers.Dropout(0.5))
    #
    model.add(tf.keras.layers.Dense(15,activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    #
    model.add(tf.keras.layers.Dense(5,activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(2, activation='softmax'))


# Multi gpu is very easy using keras
# model = tf.keras.utils.multi_gpu_model(model, gpus=2)

tensorboard=TensorBoard(log_dir='logs/{}'.format(time()))

model.compile(optimizer=tf.train.AdamOptimizer(0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset_train = dataset.batch(1024).repeat()


dataset_test=tf.data.Dataset.from_tensor_slices((x_test, y_test))
dataset_test=dataset_test.batch(1024).repeat()

model.fit(dataset_train, epochs=100, steps_per_epoch=50,callbacks=[tensorboard])

acc= model.evaluate(dataset_test,steps=50)
print('testing acc:',acc[1])

# Conclusion : IMPOSSIBLE to use DNN to predict prime ... of course
val_x=[300009,500005,333333,666666,499957,499969,499973,499979]
val=[]
for x in val_x:
    val.append(to_bin(x))
val=np.array(val)

res=model.predict(val)
for x,y in zip(val_x,res):
    print(x,' probability of prime:',y[1])