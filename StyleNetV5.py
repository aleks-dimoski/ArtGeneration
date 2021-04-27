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
num_filters = 4
batch_size = 6
learning_rate = 2e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model_name = 'V5'

conv2DT = tf.keras.layers.Conv2DTranspose
conv2D = tf.keras.layers.Conv2D


def enc_unit(inp, filter_mult=1, name='enc', first=False):
    if first:
        x = conv2D(num_filters*filter_mult*2, kernel_size=(7, 7), strides=(2, 2), padding='same')(inp)
        x = tfa.layers.InstanceNormalization()(x)
        x = tf.keras.layers.MaxPool2D()(x)
        return tf.keras.Model(inputs=inp, outputs=x, name=name)

    x = conv2D(num_filters * filter_mult, kernel_size=(3, 3), strides=(1, 1), padding='same')(inp)
    x = tfa.layers.InstanceNormalization()(x)
    x = conv2D(num_filters * filter_mult, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)

    x1 = conv2D(num_filters * filter_mult, kernel_size=(1, 1), strides=(2, 2), padding='same')(inp)
    x = tf.keras.layers.Add()([x, x1])
    x2 = tf.keras.layers.LeakyReLU()(x)

    x = conv2D(num_filters * filter_mult, kernel_size=(3, 3), strides=(1, 1), padding='same')(x2)
    x = tfa.layers.InstanceNormalization()(x)
    x = conv2D(num_filters * filter_mult, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)

    x2 = conv2D(num_filters * filter_mult, kernel_size=(1, 1), strides=(2, 2), padding='same')(x1)
    x = tf.keras.layers.Concatenate()([x, x2])
    x = tf.keras.layers.LeakyReLU()(x)

    return tf.keras.Model(inputs=inp, outputs=x, name=name)


def dec_unit(inp, filter_mult=1, name='dec', last=False):
    x = conv2DT(num_filters * filter_mult*2, (3, 3), strides=(2, 2), padding='same')(inp)
    if not last:
        x = conv2DT(num_filters * filter_mult, (3, 3), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.LeakyReLU()(x)
    else:
        x = conv2DT(3, (7, 7), strides=(2, 2), padding='same', activation='sigmoid')(x)

    return tf.keras.Model(inputs=inp, outputs=x, name=name)


def latent(inp, latent_dims=128, name='latent'):
    x = tf.keras.layers.Flatten()(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    w = tf.keras.layers.Dense(latent_dims, activation='relu')(x)
    w_mean = tf.keras.layers.Dense(latent_dims, name='w_mean')(w)
    w_log_var = tf.keras.layers.Dense(latent_dims, name='w_log_var')(w)
    w = utils.Sampling()([w_mean, w_log_var])
    return tf.keras.Model(inputs=inp, outputs=[w, w_mean, w_log_var], name=name)


def reshape_latent(inp, out_shape, name='reshape_latent'):
    x = tf.keras.layers.Dense(4*4*4*4 * num_filters, activation='relu')(inp)
    x = tf.keras.layers.Reshape(out_shape)(x)
    x = conv2D(4*4*4*4 * num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
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
        self.enc1 = enc_unit(inp=inp1, filter_mult=2, first=True, name='E1')
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
        self.latent1 = latent(inp=inp_latent, latent_dims=420)
        print(self.latent1.output_shape[0])
        inp_reshape = tf.keras.Input(shape=(420,))
        self.reshape_l = reshape_latent(inp_reshape, out_shape=(1, 1, 4*4*4*4*num_filters))
        print(self.reshape_l.output_shape)

        out4 = tf.keras.Input(shape=(1, 1, 4*4*4*4*num_filters))
        self.dec4 = dec_unit(inp=out4, filter_mult=64, name='D4')
        print(self.dec4.output_shape)
        out3 = tf.keras.Input(shape=(4, 4, 8*4*4*num_filters))
        self.dec3 = dec_unit(inp=out3, filter_mult=16, name='D3')
        print(self.dec3.output_shape)
        out2 = tf.keras.Input(shape=(16, 16, 8*4*num_filters))
        self.dec2 = dec_unit(inp=out2, filter_mult=4, name='D2')
        print(self.dec2.output_shape)
        out1 = tf.keras.Input(shape=(64, 64, 4*num_filters))
        self.dec1 = dec_unit(inp=out1, last=True, filter_mult=1, name='D1')
        print(self.dec1.output_shape)

    def save(self, fname):
        dr = os.getcwd() + '\\'
        if not os.path.isdir(dr + fname):
            try:
                os.mkdir(dr + fname)
            except OSError:
                print("File save failed.")
        tf.keras.models.save_model(model=self.enc1, filepath=dr + fname + '\\' + self.enc1.name)
        tf.keras.models.save_model(model=self.enc2, filepath=dr + fname + '\\' + self.enc2.name)
        tf.keras.models.save_model(model=self.enc3, filepath=dr + fname + '\\' + self.enc3.name)
        tf.keras.models.save_model(model=self.enc4, filepath=dr + fname + '\\' + self.enc4.name)
        tf.keras.models.save_model(model=self.latent1, filepath=dr + fname + '\\' + self.latent1.name)
        tf.keras.models.save_model(model=self.reshape_l, filepath=dr + fname + '\\' + self.reshape_l.name)
        tf.keras.models.save_model(model=self.dec4, filepath=dr + fname + '\\' + self.dec4.name)
        tf.keras.models.save_model(model=self.dec3, filepath=dr + fname + '\\' + self.dec3.name)
        tf.keras.models.save_model(model=self.dec2, filepath=dr + fname + '\\' + self.dec2.name)
        tf.keras.models.save_model(model=self.dec1, filepath=dr + fname + '\\' + self.dec1.name)

    def load(self, fname):
        dr = os.getcwd() + '\\'
        self.enc1 = tf.keras.models.load_model(dr + fname + '\\' + self.enc1.name)
        self.enc2 = tf.keras.models.load_model(dr + fname + '\\' + self.enc2.name)
        self.enc3 = tf.keras.models.load_model(dr + fname + '\\' + self.enc3.name)
        self.enc4 = tf.keras.models.load_model(dr + fname + '\\' + self.enc4.name)
        self.latent1 = tf.keras.models.load_model(dr + fname + '\\' + self.latent1.name)
        self.reshape_l = tf.keras.models.load_model(dr + fname + '\\' + self.reshape_l.name)
        self.dec4 = tf.keras.models.load_model(dr + fname + '\\' + self.dec4.name)
        self.dec3 = tf.keras.models.load_model(dr + fname + '\\' + self.dec3.name)
        self.dec2 = tf.keras.models.load_model(dr + fname + '\\' + self.dec2.name)
        self.dec1 = tf.keras.models.load_model(dr + fname + '\\' + self.dec1.name)

    #@tf.function
    def call(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        w = self.latent1(x4)[0]
        w = self.reshape_l(w)
        y4 = self.dec4(w)
        y4 = tf.keras.layers.Concatenate()([x3, y4])
        y4 = tf.keras.layers.BatchNormalization()(y4)
        y3 = self.dec3(y4)
        y3 = tf.keras.layers.Concatenate()([x2, y3])
        y3 = tf.keras.layers.BatchNormalization()(y3)
        y2 = self.dec2(y3)
        #y2 = tf.keras.layers.Concatenate()([x1, y2])
        #y2 = tf.keras.layers.BatchNormalization()(y2)
        y1 = self.dec1(y2)
        return y1

    def call_latent(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        w = self.latent1(x4)[0]
        w = self.reshape_l(w)
        y4 = self.dec4(w)
        y4 = tf.keras.layers.Concatenate()([y4, y4])
        y4 = tf.keras.layers.BatchNormalization()(y4)
        y3 = self.dec3(y4)
        y3 = tf.keras.layers.Concatenate()([y3, y3])
        y3 = tf.keras.layers.BatchNormalization()(y3)
        y2 = self.dec2(y3)
        #y2 = tf.keras.layers.Concatenate()([y2, y2])
        #y2 = tf.keras.layers.BatchNormalization()(y2)
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
        return w, w_mean, w_log_var, x4, x3, x2, x1

    def decode(self, w, x3, x2, x1):
        w = self.reshape_l(w)
        y4 = self.dec4(w)
        y4 = tf.keras.layers.Concatenate()([x3, y4])
        y4 = tf.keras.layers.BatchNormalization()(y4)
        y3 = self.dec3(y4)
        y3 = tf.keras.layers.Concatenate()([x2, y3])
        y3 = tf.keras.layers.BatchNormalization()(y3)
        y2 = self.dec2(y3)
        #y2 = tf.keras.layers.Concatenate()([x1, y2])
        #y2 = tf.keras.layers.BatchNormalization()(y2)
        y1 = self.dec1(y2)
        return y1

    def merge(self, source, style, slice=2):
        w, _, _, _, x3, x2, x1 = self.encode(source)
        source = [w, x3, x2, x1]
        w, _, _, _, x3, x2, x1 = self.encode(style)
        style = [w, x3, x2, x1]
        merged = style[:slice] + source[slice:]
        return self.decode(merged[0], merged[1], merged[2], merged[3])

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
            if i % 190 == 0:
                self.loss['Identity'] = self.loss_identity
                self.loss['KL'] = self.kl_loss
                utils.test_model(self, source, test=True, name=model_name)
                for style in zip(image_dataset.take(1)):
                    utils.test_model(self, source, style, test=True, name=model_name)
                    break
                #act = keract.get_activations(self, source)
                #keract.display_activations(act)
            print('\n')
            image_dataset, _ = utils.create_dataset(batch_size=batch_size)
            self.save(fname)
            time.sleep(.5)
        print('Training completed in', int((time.time() - start_time) / 60), "minutes &", int(duration % 60), "seconds")

    #@tf.function
    def train_step(self, source, optimizer):
        with tf.GradientTape() as tape:
            w, w_mean, w_log_var, _, _, _, _ = self.encode(source)
            prediction = self(source[0])
            prediction_latent = self.call_latent(source[0])
            loss_identity = identity_lr * learning_rate * (
                            0.2 * tf.reduce_mean(tf.reduce_sum(cross_entropy(source[0], prediction)))
                            + tf.reduce_mean(tf.reduce_sum((source[0]-prediction)**2))
                            + 0.2 * tf.reduce_mean(tf.reduce_sum(cross_entropy(source[0], prediction_latent)))
                            + tf.reduce_mean(tf.reduce_sum((source[0]-prediction_latent)**2)))
            kl_loss = -0.5 * (1 + w_log_var - tf.square(w_mean) - tf.exp(w_log_var))
            kl_loss = kl_lr * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) * learning_rate
            loss = (loss_identity + kl_loss)

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
    utils.test_model(model, source, test=True, name=model_name)
    utils.test_model(model, source, style, test=True, name=model_name)
    break