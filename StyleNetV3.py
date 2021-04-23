import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.layers import Input
import utils

assert len(tf.config.list_physical_devices('GPU')) > 0

tf.keras.backend.clear_session()
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
# tf.compat.v1.disable_eager_execution()
set_session(sess)

identity_lr = 1
kl_lr = 1
num_epochs = 200
num_filters = 12
batch_size = 16
learning_rate = 2e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model_name='V3'


def lrelu_bn(inputs):
    lrelu = tf.keras.layers.LeakyReLU()(inputs)
    bn = tf.keras.layers.BatchNormalization()(lrelu)
    return bn


def vae(name):
    conv2DT = tf.keras.layers.Conv2DTranspose
    conv2D = tf.keras.layers.Conv2D

    inp = tf.keras.Input(shape=(256, 256, 3))

    # Encoder #
    x = conv2D(6, kernel_size=(3, 3), strides=(2, 2), padding='same')(inp)
    x = tf.keras.layers.LeakyReLU()(x)
    x2 = conv2D(num_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x2)
    x = conv2D(num_filters * 2, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x1 = conv2D(num_filters * 4, kernel_size=(3, 3), strides=(2, 2), padding='same', name='skip_con')(x)
    x = tf.keras.layers.LeakyReLU()(x1)
    x = conv2D(num_filters * 8, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2D(num_filters * 16, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2D(num_filters * 32, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Latent Space #
    x = tf.keras.layers.Flatten()(x)
    w = tf.keras.layers.Dense(256, activation='relu')(x)
    w_mean = tf.keras.layers.Dense(128, name='w_mean')(w)
    w_log_var = tf.keras.layers.Dense(128, name='w_log_var')(w)
    w = utils.Sampling()([w_mean, w_log_var])

    # Decoder #
    z = tf.keras.Input(shape=(128,))
    x = tf.keras.layers.Dense(2 * 2 * 32 * num_filters, activation='relu')(z)
    x = tf.keras.layers.Reshape((2, 2, num_filters*32))(x)

    x = conv2DT(num_filters * 16, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2DT(num_filters * 8, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2DT(num_filters * 4, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Add()([x, x1])
    x = lrelu_bn(x)
    x = conv2DT(num_filters * 2, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2DT(num_filters, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Add()([x, x2])
    x = lrelu_bn(x)
    x = conv2DT(9, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    y = conv2DT(3, (3, 3), strides=(2, 2), padding='same', activation='sigmoid')(x)

    encoder = tf.keras.models.Model(inputs=inp, outputs=[w, w_mean, w_log_var, x1, x2], name='encoder')
    decoder = tf.keras.models.Model(inputs=z, outputs=y, name='decoder')
    return encoder, decoder


class AE_A(tf.keras.Model):
    def __init__(self):
        super(AE_A, self).__init__()
        self.enc, self.dec = vae('V3')
        self.compile(optimizer=optimizer)
        self.loss_identity = []
        self.kl_loss = []
        self.loss = {}

    def save(self, fname):
        dir = os.getcwd() + '\\'
        if not os.path.isdir(dir + fname):
            try:
                os.mkdir(dir + fname)
            except OSError:
                print("File save failed.")
        tf.keras.models.save_model(model=self.enc, filepath=dir + fname + '\encoder')
        tf.keras.models.save_model(model=self.dec, filepath=dir + fname + '\decoder')

    def load(self, fname):
        dir = os.getcwd() + '\\'
        self.enc = tf.keras.models.load_model(dir + fname + '\encoder')
        self.dec = tf.keras.models.load_model(dir + fname + '\decoder')

    def call(self, x):
        w = self.enc(x)
        y = self.dec(w[0], w[2], w[3])
        return y

    def decode(self, w):
        return self.dec(w)

    def encode(self, x):
        return self.enc(x)

    def train_model(self, fname=None):
        time.sleep(5)
        try:
            self.load(fname)
            print("File load successful.")
        except Exception:
            print("File load failed.")

        image_dataset, dataset_size = utils.create_dataset()
        dataset_size = dataset_size - 1
        print("Dataset size is", dataset_size)
        print("Total number of images is", dataset_size * batch_size)

        start_time = time.time()
        print("Beginning training at", start_time)

        for i in range(num_epochs):
            print("Starting epoch {}/{}".format(i, num_epochs))
            start = time.time()
            batch_on = 0
            for source in zip(image_dataset.take(int(dataset_size / 4))):
                try:
                    loss_identity, kl_loss = self.train_step(source, optimizer)
                    if batch_on % 10 == 0:
                        print("Beginning batch #" + str(batch_on), 'out of', int(dataset_size / 4), 'of size',
                              batch_size)
                        self.loss_identity += [loss_identity]
                        self.kl_loss += [kl_loss]
                except Exception:
                    print("Batch #", batch_on, "failed. Continuing with next batch.")
                batch_on += 1
            duration = time.time() - start
            print(int(duration / 60), "minutes &", int(duration % 60), "seconds, for epoch", i)
            if i % 20 == 0:
                self.loss = {'Identity': self.loss_identity, 'KL': self.kl_loss}
                utils.test_model(self, source, num=i, name=model_name)
            print('\n')
            image_dataset, _ = utils.create_dataset()
            self.save(fname)
            time.sleep(1)
        print('Training completed in', int((time.time() - start_time) / 60), "minutes &", int(duration % 60), "seconds")

    def train_step(self, source, optimizer):
        with tf.GradientTape() as tape:
            w, w_mean, w_log_var = self.encode(source)
            prediction = self(source)
            loss_identity = tf.abs(identity_lr * tf.reduce_mean((source - prediction) ** 2))
            kl_loss = -0.5 * (1 + w_log_var - tf.square(w_mean) - tf.exp(w_log_var))
            kl_loss = kl_lr * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            loss = loss_identity + kl_loss

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_identity, kl_loss

    def build_graph(self):
        source = Input(shape=(256, 256, 3))
        return tf.keras.Model(inputs=source, outputs=self.call(source), name=model_name)


model = AE_A()
tf.keras.utils.plot_model(model.build_graph(), model_name+".png", show_shapes=True, expand_nested=True)
model.train_model(model_name)
model.load(model_name)
image_dataset, _ = utils.create_dataset()

for source, style in zip(image_dataset.take(1), image_dataset.take(1)):
    utils.test_model(model, source, style, test=True, name=model_name)
    break