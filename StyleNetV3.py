import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
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
kl_lr = .1
num_epochs = 200
num_filters = 6
batch_size = 4
learning_rate = 2e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model_name = 'V3'

conv2DT = tf.keras.layers.Conv2DTranspose
conv2D = tf.keras.layers.Conv2D


def lrelu_bn(inputs):
    lrelu = tf.keras.layers.LeakyReLU()(inputs)
    bn = tf.keras.layers.BatchNormalization()(lrelu)
    return bn


def enc_unit(inp, filter_mult=1, name='enc', first=False):
    if not first:
        x = conv2D(num_filters*filter_mult, kernel_size=(3, 3), strides=(2, 2), padding='same')(inp)
    else:
        x = conv2D(num_filters*filter_mult, kernel_size=(7, 7), strides=(2, 2), padding='same')(inp)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = conv2D(num_filters*filter_mult*2, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return tf.keras.Model(inputs=inp, outputs=x, name=name)


def dec_unit(inp, filter_mult=1, name='dec', last=False):
    x = conv2DT(num_filters * filter_mult*2, (3, 3), strides=(2, 2), padding='same')(inp)
    x = tf.keras.layers.LeakyReLU()(x)
    if not last:
        x = conv2DT(num_filters * filter_mult, (7, 7), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.LeakyReLU()(x)
    else:
        x = conv2DT(3, (3, 3), strides=(2, 2), padding='same', activation='sigmoid')(x)

    return tf.keras.Model(inputs=inp, outputs=x, name=name)


def latent(inp, latent_dims=128, name='latent'):
    x = tf.keras.layers.Flatten()(inp)
    w = tf.keras.layers.Dense(latent_dims, activation='relu')(x)
    w_mean = tf.keras.layers.Dense(latent_dims, name='w_mean')(w)
    w_log_var = tf.keras.layers.Dense(latent_dims, name='w_log_var')(w)
    w = utils.Sampling()([w_mean, w_log_var])
    return tf.keras.Model(inputs=inp, outputs=[w, w_mean, w_log_var], name=name)


def reshape_latent(inp, out_shape, name='reshape_latent'):
    x = tf.keras.layers.Dense(4*4*4*4 * num_filters, activation='relu')(inp)
    x = tf.keras.layers.Reshape(out_shape)(x)
    return tf.keras.Model(inputs=inp, outputs=x, name=name)


class AE_A(tf.keras.Model):
    def __init__(self):
        super(AE_A, self).__init__()
        self.compile(optimizer=optimizer)
        self.loss_identity = []
        self.kl_loss = []
        self.loss = {}

        inp1 = tf.keras.Input(shape=(256, 256, 3))
        print(inp1.get_shape())
        self.enc1 = enc_unit(inp=inp1, filter_mult=2, name='E1')
        print(self.enc1.output_shape)
        inp2 = tf.keras.Input(shape=(64, 64, 4*num_filters))
        self.enc2 = enc_unit(inp=inp2, filter_mult=8, name='E2')
        print(self.enc2.output_shape)
        inp3 = tf.keras.Input(shape=(16, 16, 4*4*num_filters))
        self.enc3 = enc_unit(inp=inp3, filter_mult=32, name='E3')
        print(self.enc3.output_shape)
        inp4 = tf.keras.Input(shape=(4, 4, 4*4*4*num_filters))
        self.enc4 = enc_unit(inp=inp4, filter_mult=128, name='E4')
        print(self.enc4.output_shape)

        inp_latent = tf.keras.Input(shape=(1, 1, 4*4*4*4*num_filters))
        self.latent1 = latent(inp=inp_latent, latent_dims=256)
        print(self.latent1.output_shape[0])
        inp_reshape = tf.keras.Input(shape=(256,))
        self.reshape_l = reshape_latent(inp_reshape, out_shape=(1, 1, 4*4*4*4*num_filters))
        print(self.reshape_l.output_shape)

        out4 = tf.keras.Input(shape=(1, 1, 4*4*4*4*num_filters))
        self.dec4 = dec_unit(inp=out4, filter_mult=64, name='D4')
        print(self.dec4.output_shape)
        out3 = tf.keras.Input(shape=(4, 4, 4*4*4*num_filters))
        self.dec3 = dec_unit(inp=out3, filter_mult=16, name='D3')
        print(self.dec3.output_shape)
        out2 = tf.keras.Input(shape=(16, 16, 4*4*num_filters))
        self.dec2 = dec_unit(inp=out2, filter_mult=4, name='D2')
        print(self.dec2.output_shape)
        out1 = tf.keras.Input(shape=(64, 64, 4*num_filters))
        self.dec1 = dec_unit(inp=out1, last=True, filter_mult=1, name='D1')
        print(self.dec1.output_shape)

    def save(self, fname):
        dir = os.getcwd() + '\\'
        if not os.path.isdir(dir + fname):
            try:
                os.mkdir(dir + fname)
            except OSError:
                print("File save failed.")
        tf.keras.models.save_model(model=self.enc1, filepath=dir + fname + '\\' + self.enc1.name)
        tf.keras.models.save_model(model=self.enc2, filepath=dir + fname + '\\' + self.enc2.name)
        tf.keras.models.save_model(model=self.enc3, filepath=dir + fname + '\\' + self.enc3.name)
        tf.keras.models.save_model(model=self.enc4, filepath=dir + fname + '\\' + self.enc4.name)
        tf.keras.models.save_model(model=self.dec4, filepath=dir + fname + '\\' + self.dec4.name)
        tf.keras.models.save_model(model=self.dec3, filepath=dir + fname + '\\' + self.dec3.name)
        tf.keras.models.save_model(model=self.dec2, filepath=dir + fname + '\\' + self.dec2.name)
        tf.keras.models.save_model(model=self.dec1, filepath=dir + fname + '\\' + self.dec1.name)

    def load(self, fname):
        dir = os.getcwd() + '\\'
        self.enc1 = tf.keras.models.load_model(dir + fname + '\\' + self.enc1.name)
        self.enc2 = tf.keras.models.load_model(dir + fname + '\\' + self.enc2.name)
        self.enc3 = tf.keras.models.load_model(dir + fname + '\\' + self.enc3.name)
        self.enc4 = tf.keras.models.load_model(dir + fname + '\\' + self.enc4.name)
        self.dec4 = tf.keras.models.load_model(dir + fname + '\\' + self.dec4.name)
        self.dec3 = tf.keras.models.load_model(dir + fname + '\\' + self.dec3.name)
        self.dec2 = tf.keras.models.load_model(dir + fname + '\\' + self.dec2.name)
        self.dec1 = tf.keras.models.load_model(dir + fname + '\\' + self.dec1.name)

    #@tf.function
    def call(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        w = self.latent1(x4)[0]
        w = self.reshape_l(w)
        y4 = self.dec4(w)
        y4 = tf.keras.layers.Add()([x3, y4])
        y4 = lrelu_bn(y4)
        y3 = self.dec3(y4)
        y3 = tf.keras.layers.Add()([x2, y3])
        y3 = lrelu_bn(y3)
        y2 = self.dec2(y3)
        y2 = tf.keras.layers.Add()([x1, y2])
        y2 = lrelu_bn(y2)
        y1 = self.dec1(y2)
        return y1

    def encode(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        w_hold = self.latent1(x4)
        w = w_hold[0]
        w_mean = w_hold[1]
        w_log_var = w_hold[2]
        return w, w_mean, w_log_var

    def decode(self, w, x1, x2, x3):
        w = self.reshape_l(w)
        y4 = self.dec4(w)
        y4 = tf.keras.layers.Add()([x3, y4])
        y4 = lrelu_bn(y4)
        y3 = self.dec3(y4)
        y3 = tf.keras.layers.Add()([x2, y3])
        y3 = lrelu_bn(y3)
        y2 = self.dec2(y3)
        y2 = tf.keras.layers.Add()([x1, y2])
        y2 = lrelu_bn(y2)
        y1 = self.dec1(y2)
        return y1

    #@tf.function
    def train_model(self, fname=None):
        time.sleep(1)
        try:
            self.load(fname)
            print("File load successful.")
        except Exception:
            print("File load failed.")

        image_dataset, dataset_size = utils.create_dataset(batch_size=batch_size)
        dataset_size = dataset_size - 1
        print("Dataset size is", dataset_size)
        print("Total number of images is", dataset_size * batch_size)
        record_batch = utils.record_steps(int(dataset_size/2))

        start_time = time.time()
        print("Beginning training at", start_time)

        for i in range(num_epochs):
            print("Starting epoch {}/{}".format(i, num_epochs))
            start = time.time()
            batch_on = 0
            for source in zip(image_dataset.take(int(dataset_size / 2))):
                try:
                    loss_identity, kl_loss = self.train_step(source, optimizer)
                    if batch_on % record_batch == 0:
                        print("Beginning batch #" + str(batch_on), 'out of', int(dataset_size / 2), 'of size', batch_size)
                        self.loss_identity += [loss_identity]
                        self.kl_loss += [kl_loss]
                except Exception:
                    print("Batch #", batch_on, "failed. Continuing with next batch.")
                batch_on += 1
            duration = time.time() - start
            print(int(duration / 60), "minutes &", int(duration % 60), "seconds, for epoch", i)
            if i % 10 == 0:
                self.loss['Identity'] = self.loss_identity
                self.loss['KL'] = self.kl_loss
                utils.test_model(self, source, num=i, name=model_name)
            if i % 10 == 0:
                self.loss['Identity'] = self.loss_identity
                self.loss['KL'] = self.kl_loss
                utils.test_model(self, source, test=True, name=model_name)
                #act = keract.get_activations(self, source)
                #keract.display_activations(act)
            print('\n')
            time.sleep(.1)
            image_dataset, _ = utils.create_dataset(batch_size=batch_size)
            self.save(fname)
        print('Training completed in', int((time.time() - start_time) / 60), "minutes &", int(duration % 60), "seconds")

    #@tf.function
    def train_step(self, source, optimizer):
        with tf.GradientTape() as tape:
            w, w_mean, w_log_var = self.encode(source)
            prediction = self(source[0])
            loss_identity = identity_lr * tf.reduce_mean(tf.reduce_sum(cross_entropy(source[0], prediction))) #+ 0.1 * tf.reduce_mean((source[0]-prediction)**2))
            kl_loss = -0.5 * (1 + w_log_var - tf.square(w_mean) - tf.exp(w_log_var))
            kl_loss = kl_lr * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            loss = (loss_identity + kl_loss) * learning_rate

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
image_dataset, _ = utils.create_dataset(batch_size=batch_size)

for source, style in zip(image_dataset.take(1), image_dataset.take(1)):
    utils.test_model(model, source, style, test=True, name=model_name)
    break