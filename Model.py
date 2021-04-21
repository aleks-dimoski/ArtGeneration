import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.layers import Input
from PIL import Image

assert len(tf.config.list_physical_devices('GPU')) > 0

tf.keras.backend.clear_session()
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
#tf.compat.v1.disable_eager_execution()
set_session(sess)

content_lr = 0.8
style_lr = 1
identity_lr = 2
num_epochs = 200
num_filters = 12
batch_size = 16
learning_rate = 2e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def lrelu_bn(inputs):
    lrelu = tf.keras.layers.LeakyReLU()(inputs)
    bn = tf.keras.layers.BatchNormalization()(lrelu)
    return bn


def enc(name):
    conv2D = tf.keras.layers.Conv2D
    normalize = tfa.layers.InstanceNormalization
    inp = tf.keras.Input(shape=(256, 256, 3))
    x = tfa.layers.InstanceNormalization()(inp)
    x = conv2D(6, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x2 = conv2D(num_filters, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.LeakyReLU()(x2)
    x = conv2D(num_filters*2, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x1 = conv2D(num_filters*4, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', name='skip_con')(x)
    x = tf.keras.layers.LeakyReLU()(x1)
    x = conv2D(num_filters*8, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2D(num_filters*16, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2D(num_filters*32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.LeakyReLU()(x)

    ### Latent Space ###
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    w = tf.keras.layers.Dense(32*2*2*num_filters, activation='relu')(x)
    #x = z#tf.keras.layers.Dense(1, activation='sigmoid')(z)

    model = tf.keras.models.Model(inputs=inp, outputs=[w, x1, x2], name=name)
    return model


def dec(name):
    conv2DT = tf.keras.layers.Conv2DTranspose
    normalize = tf.keras.layers.BatchNormalization
    inp = tf.keras.Input(shape=(32*2*2*num_filters))
    skip_con1 = tf.keras.Input(shape=(16, 16, 48))
    skip_con2 = tf.keras.Input(shape=(64, 64, 12))
    x = tf.keras.layers.Reshape((2, 2, 32 * num_filters))(inp)

    x = conv2DT(num_filters*16, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2DT(num_filters*8, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2DT(num_filters*4, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Add()([x, skip_con1])
    x = lrelu_bn(x)
    x = conv2DT(num_filters*2, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2DT(num_filters, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Add()([x, skip_con2])
    x = lrelu_bn(x)
    x = conv2DT(6, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    w = conv2DT(3, (3, 3), strides=(2, 2), padding='same', activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=[inp, skip_con1, skip_con2], outputs=w, name=name)
    return model


def create_dataset(image_paths=None):
    dgen_params = dict(
        rescale=1./255,
        shear_range=0.0,
        zoom_range=0.0,
        horizontal_flip=True,
        brightness_range=(0.7, 1.2),
    )
    gen_params = dict(
        batch_size=batch_size,
        color_mode='rgb',
        class_mode=None
    )
    image_dgen = tf.keras.preprocessing.image.ImageDataGenerator(**dgen_params)
    image_gen = image_dgen.flow_from_directory('D:\Storage\Technical\Linux Resources\Images\ArtGen',**gen_params)
    image_dataset = tf.data.Dataset.from_generator(lambda: image_gen, output_signature=tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32))
    return image_dataset, len(image_gen)


def test_model(model, source, style, num=0):
    _, _, new_img = model(source, style)
    pred = Image.fromarray(np.array(new_img[0]), 'RGB')
    pred.save(os.path.join('pred', 'pred_at_epoch_'+str(num)+'.png'))
    '''
    plt.figure(figsize=(6, 6))

    plt.subplot(3, 2, 1)
    plt.imshow(source[0])
    plt.grid(False)

    plt.subplot(3, 2, 2)
    plt.imshow(style[0])
    plt.grid(False)

    plt.subplot(3, 1, 2)
    
    plt.imshow(pred)
    plt.grid(False)

    plt.subplot(3, 1, 3)
    indices = [i for i in range(len(model.loss_content))]
    plt.plot(indices, model.loss_content, label='Content', color='blue')
    plt.plot(indices, model.loss_style, label='Style', color='green')
    plt.plot(indices, model.loss_identity, label='Identity', color='red')
    plt.legend()
    plt.grid(False)

    plt.show()'''


class AE_A(tf.keras.Model):
    def __init__(self):
        super(AE_A, self).__init__()
        self.encoder1 = enc("enc1")
        self.encoder2 = enc("enc2")
        self.decoder = dec("dec")
        self.compile(optimizer=optimizer)
        self.loss_content = []
        self.loss_style = []
        self.loss_identity = []

    def save(self, fname):
        dir = os.getcwd() + '\\'
        if not os.path.isdir(dir + fname):
            try:
                os.mkdir(dir + fname)
            except OSError:
                print("File save failed.")
        tf.keras.models.save_model(model=self.encoder1, filepath=dir + fname + '\encoder1')
        tf.keras.models.save_model(model=self.encoder2, filepath=dir + fname + '\encoder2')
        tf.keras.models.save_model(model=self.decoder, filepath=dir + fname + '\decoder')

    def load(self, fname):
        dir = os.getcwd() + '\\'
        self.encoder1 = tf.keras.models.load_model(dir + fname + '\encoder1')
        self.encoder2 = tf.keras.models.load_model(dir + fname + '\encoder2')
        self.decoder = tf.keras.models.load_model(dir + fname + '\decoder')

    def call(self, source, style):
        source_encoded = self.encoder1(source)
        style_encoded = self.encoder2(style)
        encoded = tf.keras.layers.Add(name='combine')([source_encoded[0], style_encoded[0]])
        decoded = self.decoder([encoded, source_encoded[1], source_encoded[2]])
        return source_encoded, style_encoded, decoded

    def train_model(self, fname=None):
        time.sleep(5)
        try:
            self.load(fname)
            print("File load successful.")
        except Exception:
            print("File load failed.")

        image_dataset, dataset_size = create_dataset()
        dataset_size = dataset_size-1
        print("Dataset size is", dataset_size)
        print("Total number of images is", dataset_size*batch_size)

        start_time = time.time()
        print("Beginning training at", start_time)
        for i in range(num_epochs):
            print("Starting epoch {}/{}".format(i, num_epochs))
            start = time.time()
            batch_on = 0
            for source, style in zip(image_dataset.take(int(dataset_size/4)), image_dataset.take(int(dataset_size/4))):
                try:
                    loss_content, loss_style, loss_identity = self.train_step(source, style, optimizer)
                    if batch_on % 10 == 0:
                        print("Beginning batch #" + str(batch_on), 'out of', int(dataset_size/4), 'of size', batch_size)
                        self.loss_content += [loss_content]
                        self.loss_style += [loss_style]
                        self.loss_identity += [loss_identity]
                except Exception:
                    print("Batch #", batch_on, "failed. Continuing with next batch.")
                batch_on += 1
            duration = time.time()-start
            print(int(duration/60), "minutes &", int(duration % 60), "seconds, for epoch", i)
            if i % 20 == 0:
                test_model(self, source, style, i)
            print('\n')
            image_dataset, _ = create_dataset()
            self.save("test")
            time.sleep(1)
        print('Training completed in', int((time.time()-start_time) / 60), "minutes &", int(duration % 60), "seconds")

    def train_step(self, source, style, optimizer):
        with tf.GradientTape() as tape:
            encoder2 = self.encoder2
            _, _, prediction = self(source, style)
            _, _, source_reconstruction = self(source, source)
            # _, _, style_reconstruction = self(style, style)
            loss_content = tf.abs(content_lr * tf.reduce_mean((source-prediction)**2))
            loss_style = tf.abs(style_lr * tf.reduce_mean((encoder2(style)[0]-encoder2(prediction)[0])**2))
            loss_identity = tf.abs(identity_lr * tf.reduce_mean((source-source_reconstruction)**2))
            loss = loss_content + loss_style + loss_identity

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_content, loss_style, loss_identity

    def build_graph(self):
        source = Input(shape=(256, 256, 3))
        style = Input(shape=(256, 256, 3))
        return tf.keras.Model(inputs=[source, style], outputs=self.call(source, style))


model = AE_A()
tf.keras.utils.plot_model(model.build_graph(), "test.png", show_shapes=True, expand_nested=True)
model.train_model('test')
model.load('test')
image_dataset, _ = create_dataset()

for source, style in zip(image_dataset.take(1), image_dataset.take(1)):
    test_model(model, source, style)
    break