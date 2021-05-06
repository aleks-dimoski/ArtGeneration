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
kl_lr = .5
num_epochs = 200
num_filters = 1
batch_size = 2
learning_rate = 2e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model_name = 'V6'
display_mod = 10

conv2DT = tf.keras.layers.Conv2DTranspose
conv2D = tf.keras.layers.Conv2D


def res_block(inp, filter_mult, reduce_size=False):
    x = conv2D(num_filters * filter_mult, kernel_size=(1, 1), strides=(1, 1), padding='same')(inp)
    x = tf.keras.activations.gelu(x)
    x = conv2D(num_filters * filter_mult, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.activations.gelu(x)
    x = conv2D(num_filters * filter_mult, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.activations.gelu(x)
    x = conv2D(num_filters * filter_mult, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.activations.gelu(x)
    x = tf.keras.layers.Add()([x, inp])
    return x


def enc_unit(inp, filter_mult=1, name='enc', first=False):
    if first:
        x = conv2D(num_filters*filter_mult, kernel_size=(7, 7), strides=(2, 2), padding='same')(inp)
        x = tfa.layers.InstanceNormalization()(x)
        x = tf.keras.layers.AvgPool2D()(x)

        x = res_block(x, filter_mult)
        x = res_block(x, filter_mult)
        x = res_block(x, filter_mult)
        x = conv2D(num_filters * filter_mult * 4, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.AvgPool2D()(x)

        return tf.keras.Model(inputs=inp, outputs=x, name=name)

    x = res_block(inp, filter_mult)
    x = res_block(x, filter_mult)
    x = res_block(x, filter_mult)
    x = conv2D(num_filters*filter_mult*4, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.AvgPool2D()(x)

    return tf.keras.Model(inputs=inp, outputs=x, name=name)


def dec_unit(inp, filter_mult, name='dec'):
    x1 = res_block(inp[0], filter_mult=filter_mult)
    x2 = conv2D(num_filters*filter_mult, kernel_size=(1, 1), strides=(1, 1), padding='same')(inp[1])
    x = tf.keras.layers.Add()([inp[0], x1])
    x = tf.keras.layers.Add()([x, x2])
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    return tf.keras.Model(inputs=inp, outputs=x, name=name)


'''def dec_unit(inp, filter_mult=1, name='dec', last=False, first=False):
    #x = conv2DT(num_filters * filter_mult*2, (3, 3), strides=(2, 2), padding='same')(inp)
    if first and not last:
        x = conv2DT(num_filters * filter_mult, (1, 1), strides=(1, 1), padding='same')(inp)
        x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        x = tf.keras.layers.LeakyReLU()(x)
        return tf.keras.Model(inputs=inp, outputs=x, name=name)
    if not last:
        x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(inp)
        x = conv2DT(num_filters * filter_mult, (3, 3), strides=(1, 1), padding='same')(x)
        return tf.keras.Model(inputs=inp, outputs=x, name=name)
    else:
        x = conv2DT(64, (1, 1), strides=(2, 2), padding='same')(inp)
        x = tf.keras.layers.LeakyReLU()(x)
        x = conv2DT(3, (7, 7), strides=(2, 2), padding='same', activation='sigmoid')(x)
        return tf.keras.Model(inputs=inp, outputs=x, name=name)'''


def latent_skip(x, latent_dims=16, name='latent_skip', training=True):
    filter_count = x.get_shape().as_list()[-1]
    y = tf.keras.layers.Conv2D(filter_count, (1, 1), strides=(1, 1), padding='same')(x)
    return tf.keras.Model(inputs=x, outputs=y, name=name)


def latent(inp, latent_dims=128, name='latent', training=True):
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same')(inp)
    w = tf.keras.layers.Flatten()(x)
    w = tf.keras.layers.Dense(latent_dims)(w)
    if training:
        w = tf.keras.layers.Dropout(0.4)(w)
    w_mean = tf.keras.layers.Dense(latent_dims, name='w_mean')(w)
    w_log_var = tf.keras.layers.Dense(latent_dims, name='w_log_var')(w)
    w = utils.Sampling()([w_mean, w_log_var])
    w = tf.keras.layers.Dropout(0.05)(w)
    return tf.keras.Model(inputs=inp, outputs=[w, w_mean, w_log_var], name=name)


def reshape_latent(inp, filter_mult, out_shape, name='reshape_latent'):
    x = tf.keras.layers.Dense(utils.get_size(out_shape), activation='relu')(inp)
    x = tf.keras.layers.Reshape(out_shape)(x)
    x = tf.keras.layers.Conv2D(num_filters*filter_mult, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    return tf.keras.Model(inputs=inp, outputs=x, name=name)


class AE_A(tf.keras.Model):
    def __init__(self, training=True):
        super(AE_A, self).__init__()
        self.compile(optimizer=optimizer)
        self.loss_identity = []
        self.kl_loss = []
        self.loss = {}

        inp1 = tf.keras.Input(shape=(256, 256, 3))
        print(inp1.get_shape())
        self.enc1 = enc_unit(inp=inp1, filter_mult=3, first=True, name='E1')
        print(self.enc1.output_shape)
        inp2 = tf.keras.Input(shape=(self.enc1.output_shape[1:]))
        self.enc2 = enc_unit(inp=inp2, filter_mult=4*3, name='E2')
        print(self.enc2.output_shape)
        inp3 = tf.keras.Input(shape=(self.enc2.output_shape[1:]))
        self.enc3 = enc_unit(inp=inp3, filter_mult=(4**2)*3, name='E3')
        print(self.enc3.output_shape)
        inp4 = tf.keras.Input(shape=(self.enc3.output_shape[1:]))
        self.enc4 = enc_unit(inp=inp4, filter_mult=(4**3)*3, name='E4')
        print(self.enc4.output_shape)
        inp5 = tf.keras.Input(shape=(self.enc4.output_shape[1:]))
        self.enc5 = enc_unit(inp=inp5, filter_mult=(4**4)*3, name='E5')
        print(self.enc5.output_shape)

        '''inp_latent = tf.keras.Input(shape=(self.enc5.output_shape[1:]))
        self.latent1 = latent(inp=inp_latent, latent_dims=600*num_filters, training=training)
        print('latent', self.latent1.output_shape[0])
        inp_reshape = tf.keras.Input(shape=(600*num_filters))
        self.reshape_l = reshape_latent(inp_reshape, out_shape=(self.enc5.output_shape[1:]))
        print(self.reshape_l.output_shape)'''

        self.skip4 = latent_skip(inp5, name='latent_skip4', training=training) #256
        self.skip3 = latent_skip(inp4, name='latent_skip3', training=training) #128
        self.skip2 = latent_skip(inp3, name='latent_skip2', training=training) #32
        self.skip1 = latent_skip(inp2, name='latent_skip1', training=training) #16

        out5 = tf.keras.Input(shape=(self.enc5.output_shape[1:]))
        self.dec5 = dec_unit(inp=out5, filter_mult=(4**4)*3, name='D5')
        print(self.dec5.output_shape)
        out4 = tf.keras.Input(shape=(self.dec5.output_shape[1:]))
        self.dec4 = dec_unit(inp=out4, filter_mult=(4**3)*3, name='D4')
        print(self.dec4.output_shape)
        out3 = tf.keras.Input(shape=(self.dec4.output_shape[1:]))
        self.dec3 = dec_unit(inp=out3, filter_mult=(4**2)*3, name='D3')
        print(self.dec3.output_shape)
        out2 = tf.keras.Input(shape=(self.dec3.output_shape[1:]))
        self.dec2 = dec_unit(inp=out2, filter_mult=4*3, name='D2')
        print(self.dec2.output_shape)
        out1 = tf.keras.Input(shape=(self.dec2.output_shape[1:]))
        self.dec1 = dec_unit(inp=out1, last=True, filter_mult=3, name='D1')
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
        tf.keras.models.save_model(model=self.enc5, filepath=dr + fname + '\\' + self.enc5.name)
        #tf.keras.models.save_model(model=self.latent1, filepath=dr + fname + '\\' + self.latent1.name)
        #tf.keras.models.save_model(model=self.reshape_l, filepath=dr + fname + '\\' + self.reshape_l.name)
        tf.keras.models.save_model(model=self.skip4, filepath=dr + fname + '\\' + self.skip4.name)
        tf.keras.models.save_model(model=self.skip3, filepath=dr + fname + '\\' + self.skip3.name)
        tf.keras.models.save_model(model=self.skip2, filepath=dr + fname + '\\' + self.skip2.name)
        tf.keras.models.save_model(model=self.skip1, filepath=dr + fname + '\\' + self.skip1.name)
        tf.keras.models.save_model(model=self.dec5, filepath=dr + fname + '\\' + self.dec5.name)
        tf.keras.models.save_model(model=self.dec4, filepath=dr + fname + '\\' + self.dec4.name)
        tf.keras.models.save_model(model=self.dec3, filepath=dr + fname + '\\' + self.dec3.name)
        tf.keras.models.save_model(model=self.dec2, filepath=dr + fname + '\\' + self.dec2.name)
        tf.keras.models.save_model(model=self.dec1, filepath=dr + fname + '\\' + self.dec1.name)

    def load(self, fname, compile=True):
        dr = os.getcwd() + '\\'
        self.enc1 = tf.keras.models.load_model(dr + fname + '\\' + self.enc1.name, compile=compile)
        self.enc2 = tf.keras.models.load_model(dr + fname + '\\' + self.enc2.name, compile=compile)
        self.enc3 = tf.keras.models.load_model(dr + fname + '\\' + self.enc3.name, compile=compile)
        self.enc4 = tf.keras.models.load_model(dr + fname + '\\' + self.enc4.name, compile=compile)
        self.enc5 = tf.keras.models.load_model(dr + fname + '\\' + self.enc5.name, compile=compile)
        #self.latent1 = tf.keras.models.load_model(dr + fname + '\\' + self.latent1.name, compile=compile)
        #self.reshape_l = tf.keras.models.load_model(dr + fname + '\\' + self.reshape_l.name, compile=compile)
        self.skip4 = tf.keras.models.load_model(dr + fname + '\\' + self.skip4.name, compile=compile)
        self.skip3 = tf.keras.models.load_model(dr + fname + '\\' + self.skip3.name, compile=compile)
        self.skip2 = tf.keras.models.load_model(dr + fname + '\\' + self.skip2.name, compile=compile)
        self.skip1 = tf.keras.models.load_model(dr + fname + '\\' + self.skip1.name, compile=compile)
        self.dec5 = tf.keras.models.load_model(dr + fname + '\\' + self.dec5.name, compile=compile)
        self.dec4 = tf.keras.models.load_model(dr + fname + '\\' + self.dec4.name, compile=compile)
        self.dec3 = tf.keras.models.load_model(dr + fname + '\\' + self.dec3.name, compile=compile)
        self.dec2 = tf.keras.models.load_model(dr + fname + '\\' + self.dec2.name, compile=compile)
        self.dec1 = tf.keras.models.load_model(dr + fname + '\\' + self.dec1.name, compile=compile)

    #@tf.function
    def call(self, x):
        y1 = self.merge(x, x)
        return y1

    def encode(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        return x5, x4, x3, x2, x1

    def decode(self, x5, x4, x3, x2, x1):
        x4 = self.skip4(x4)
        x3 = self.skip3(x3)
        x2 = self.skip2(x2)
        x1 = self.skip1(x1)
        y5 = self.dec5(x5)
        y5 = tf.keras.layers.Add()([x4, y5])
        y5 = tf.keras.layers.BatchNormalization()(y5)
        y4 = self.dec4(y5)
        y4 = tf.keras.layers.Add()([x3, y4])
        y4 = tf.keras.layers.BatchNormalization()(y4)
        y3 = self.dec3(y4)
        y3 = tf.keras.layers.Add()([x2, y3])
        y3 = tf.keras.layers.BatchNormalization()(y3)
        y2 = self.dec2(y3)
        y2 = tf.keras.layers.Add()([x1, y2])
        y2 = tf.keras.layers.BatchNormalization()(y2)
        y1 = self.dec1(y2)
        return y1

    def merge(self, source, style, slice=2):
        x5, x4, x3, x2, x1 = self.encode(source)
        source = [x5, x4, x3, x2, x1]
        x5, x4, x3, x2, x1 = self.encode(style)
        style = [x5, x4, x3, x2, x1]
        merged = style[:slice] + source[slice:]
        return self.decode(merged[0], merged[1], merged[2], merged[3], merged[4])

    #@tf.function
    def train_model(self, fname=None):
        time.sleep(1)
        try:
            self.load(fname)
            print("File load successful.")
        except Exception:
            print("File load failed.")

        model_val = AE_A(training=False)

        image_dataset, dataset_size = utils.create_dataset(batch_size=batch_size)
        dataset_size = int((dataset_size - 1)/2)
        print("Dataset size is", dataset_size)
        print("Total number of images is", dataset_size * batch_size)
        record_batch = utils.record_steps(dataset_size)

        start_time = time.time()
        print("Beginning training at", start_time)

        for i in range(num_epochs):
            print(f'Starting epoch {i}/{num_epochs}')
            start = time.time()
            batch_on = 0

            for source in zip(image_dataset.take(dataset_size)):
                try:
                    loss_identity = self.train_step(source, optimizer)
                    if batch_on % record_batch == 0:
                        print(f'Beginning batch #{batch_on} out of {dataset_size} of size {batch_size}')
                        self.loss_identity += [loss_identity]
                        self.kl_loss += [0]
                except Exception:
                    print(f'Batch #{batch_on} failed. Continuing with next batch.')
                batch_on += 1
            self.save(fname)
            time.sleep(0.5)
            duration = time.time() - start
            utils.print_time_remaining(i, num_epochs, duration)

            if i % display_mod == 0:
                model_val.load(model_name, compile=False)
                model_val.loss['Identity'] = self.loss_identity
                model_val.loss['KL'] = self.kl_loss
                utils.test_model(model_val, source, num=i, test=True, name=model_name, details='identity')
                for style in zip(image_dataset.take(1)):
                    utils.test_model(model_val, source, style, num=i, test=True, name=model_name, details='merge')
                    break
                #act = keract.get_activations(self, source)
                #keract.display_activations(act)
            print('\n')

            image_dataset, _ = utils.create_dataset(batch_size=batch_size)
        print('Training completed in', int((time.time() - start_time) / 60), "minutes &", int(duration % 60), "seconds")

    #@tf.function
    def train_step(self, source, optimizer):
        with tf.GradientTape() as tape:
            prediction = self(source[0])
            loss_identity = identity_lr * (
                            0.5 * tf.reduce_mean(tf.reduce_sum(cross_entropy(source[0], prediction)))
                            + 0.5 * tf.reduce_mean(tf.reduce_sum((source[0]-prediction)**2)))
            loss = loss_identity

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_identity

    def build_graph(self):
        source = Input(shape=(256, 256, 3))
        network = tf.keras.Model(inputs=source, outputs=self.call(source), name=model_name)
        print(network.summary())
        return network


model = AE_A()
tf.keras.utils.plot_model(model.build_graph(), model_name+'.png', show_shapes=True, expand_nested=True)
model.train_model(model_name)
model.load(model_name)
model.trainable = False
image_dataset, length = utils.create_dataset(batch_size=batch_size)

for source in zip(image_dataset.take(1)):
    for style in zip(image_dataset.take(1)):
        utils.test_model(model, source, test=True, name=model_name)
        utils.test_model(model, source, style, test=True, name=model_name)
        break
    break