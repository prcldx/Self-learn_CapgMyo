import tensorflow as tf
import numpy as np
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection

import get_data
import Pre_process
#建立模型


net=tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=16,kernel_size=4,activation='sigmoid',padding='same',input_shape=(16,8,1)),
    tf.keras.layers.MaxPool2D(pool_size=2,strides=2,padding='same'),
    tf.keras.layers.Conv2D(filters=16,kernel_size=2,activation='sigmoid'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120,activation='sigmoid'),
    tf.keras.layers.Dense(84,activation='sigmoid'),
    tf.keras.layers.Dense(6,activation='softmax')])

X = tf.random.uniform((1,16,8,1))
for layer in net.layers:
    X = layer(X)
    print(layer.name, 'output shape\t', X.shape)
