import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.python.keras.backend import set_session
import functools
from tensorflow.keras.layers import Input, Concatenate

assert len(tf.config.list_physical_devices('GPU')) > 0

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
set_session(sess)

num_epochs = 5
num_filters = 16
batch_size = 16
learning_rate = 2e-2
latent_dim = 30

def encoder():
    conv2D = tf.keras.layers.Conv2D
    normalize = tfa.keras.layers.InstanceNormalization
    seq = tf.keras.Sequential([
        conv2D(num_filters, (3, 3), strides=2, activation='relu'),
        normalize(),
        conv2D(num_filters, (3, 3), strides=2, activation='relu'),
        normalize(),
    ])
    return seq

def enc():
    x = Input(shape=(3,256,256))

    ### Layer 0 ###
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(7,7), activation = 'relu')(x)
    w = tfa.keras.layers.InstanceNormalization()(x)
    x = tf.keras.layers.MaxPool2D((2,2))

    ### Layer 1 ###
    y = encoder()(x)
    w = Concatenate()([x,y])
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), activation = 'relu')(w)
    y = encoder()(x)
    w = Concatenate()([x, y])
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), activation = 'relu')(w)

    ### Layer 2 ###
    y = encoder()(x)
    w = Concatenate()([x, y])
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), strides=2, activation = 'relu')(w)
    y = encoder()(x)
    w = Concatenate()([x, y])
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), activation = 'relu')(w)

    ### Layer 3 ###
    y = encoder()(x)
    w = Concatenate()([x, y])
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), strides=2, activation = 'relu')(w)
    y = encoder()(x)
    w = Concatenate()([x, y])
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), activation = 'relu')(w)

    ### Layer 4 ###
    y = encoder()(x)
    w = Concatenate()([x, y])
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), strides=2, activation = 'relu')(w)
    y = encoder()(x)
    w = Concatenate()([x, y])
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), activation = 'relu')(w)

    model = Model(inputs = Input(shape=(3,256,256)), outputs = x, name='Encoder1')
    return model

def dec():
    conv2DT = functools.partial(tf.keras.layers.Conv2DTranspose, padding='same', activation='relu')
    normalize = tf.keras.layers.BatchNormalization
    x = Input(shape=(3,256,256))
    z = enc()(x)
    inShape = tf.shape(z)


    ### ????? ###
    x = conv2DT(num_filters, (3, 3), strides=2, activation='relu')(inShape)
    w = normalize()(x)
    x = conv2DT(num_filters, (3, 3), strides=2, activation='relu')(w)
    w = normalize()(x)
    x = conv2DT(num_filters, (3, 3), strides=2, activation='relu')(w)
    w = normalize()(x)
    x = conv2DT(num_filters, (3, 3), strides=2, activation='relu')(w)
    w = normalize()(x)
    x = conv2DT(1, (3, 3), strides=2, activation='relu')(w)

class Model():
    def __init__(self):
        self.encoder = encoder()

    def save(self, fname):
        dir = os.getcwd()