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
    x = Input(shape=(128, 128, 3))

    ### Layer 0 ###
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(7, 7), activation='relu')(x)
    w = tfa.keras.layers.InstanceNormalization()(x)
    x = tf.keras.layers.MaxPool2D((2, 2))

    ### Layer 1 ###
    y = encoder()(x)
    w = Concatenate()([x, y])
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), activation='relu')(w)
    y = encoder()(x)
    w = Concatenate()([x, y])
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), activation='relu')(w)

    ### Layer 2 ###
    y = encoder()(x)
    w = Concatenate()([x, y])
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), strides=2, activation='relu')(w)
    y = encoder()(x)
    w = Concatenate()([x, y])
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), activation='relu')(w)

    ### Layer 3 ###
    y = encoder()(x)
    w = Concatenate()([x, y])
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), strides=2, activation='relu')(w)
    y = encoder()(x)
    w = Concatenate()([x, y])
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), activation='relu')(w)

    ### Layer 4 ###
    y = encoder()(x)
    w = Concatenate()([x, y])
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), strides=2, activation='relu')(w)
    y = encoder()(x)
    w = Concatenate()([x, y])
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), activation='relu')(w)

    ### Classification ###
    y = tf.keras.layers.Flatten()(x)
    z = tf.keras.layers.Dense(512, activation='relu')(y)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(z)

    model = Model(inputs=Input(shape=(128, 128, 3)), outputs=x, name='Encoder1')
    return model


def dec():
    conv2DT = functools.partial(tf.keras.layers.Conv2DTranspose, padding='same', activation='relu')
    normalize = tf.keras.layers.BatchNormalization
    w = Input(shape=(52, 52, 3))

    ### ????? ###
    x = conv2DT(num_filters, (3, 3), strides=2, activation='relu')(w)
    w = normalize()(x)
    x = conv2DT(num_filters, (3, 3), strides=2, activation='relu')(w)
    w = normalize()(x)
    x = conv2DT(num_filters, (3, 3), strides=2, activation='relu')(w)
    w = normalize()(x)
    x = conv2DT(num_filters, (3, 3), strides=2, activation='relu')(w)
    w = normalize()(x)
    x = conv2DT(1, (3, 3), strides=2, activation='relu')(w)
    w = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense((3 * 128 * 128), activation='relu')(w)
    w = tf.keras.layers.Reshape(target_shape=(128, 128, 3))(x)

    model = Model(inputs=Input(shape=(52, 52, 3)), outputs=w, name='Decoder1')
    return model


def train_model():
    img_paths = []
    for dir in os.listdir('true_art'):
        names = [(dir + '/' + x) for x in os.listdir(dir)]
        img_paths += names


class Model:
    def __init__(self):
        self.encoder = enc()
        self.decoder = dec()

    def save(self, fname):
        dir = os.getcwd()

    def train(self):
        train_model()
        return None

