import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.layers import Input, Concatenate
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

assert len(tf.config.list_physical_devices('GPU')) > 0

tf.keras.backend.clear_session()
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
#tf.compat.v1.disable_eager_execution()
set_session(sess)

reconstruction_learning_rate = 0.4
num_epochs = 35
num_filters = 12
batch_size = 4
learning_rate = 5e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def encoder():
    conv2D = tf.keras.layers.Conv2D
    normalize = tfa.layers.InstanceNormalization
    seq = tf.keras.Sequential([
        conv2D(num_filters, (3, 3), strides=2, padding='same', activation='relu'),
        normalize(),
        conv2D(num_filters, (3, 3), strides=2, padding='same', activation='relu'),
        normalize(),
    ])
    return seq


def enc(name):
    conv2D = tf.keras.layers.Conv2D
    normalize = tfa.layers.InstanceNormalization
    inp = tf.keras.Input(shape=(256, 256, 3))
    x = tfa.layers.InstanceNormalization()(inp)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)

    x = conv2D(num_filters, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2D(num_filters*2, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2D(num_filters*4, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2D(num_filters*8, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2D(num_filters*16, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.LeakyReLU()(x)

    ### Latent Space ###
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512/2, activation='relu')(x)
    w = tf.keras.layers.Dense(32*4*4*num_filters/2, activation='relu')(x)
    #x = z#tf.keras.layers.Dense(1, activation='sigmoid')(z)

    model = tf.keras.models.Model(inputs=inp, outputs=w, name=name)
    return model


def dec(name):
    conv2DT = tf.keras.layers.Conv2DTranspose
    normalize = tf.keras.layers.BatchNormalization
    inp = tf.keras.Input(shape=(32*4*4*num_filters))
    x = tf.keras.layers.Reshape((4, 4, 32 * num_filters))(inp)

    x = conv2DT(num_filters*32, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2DT(num_filters*16, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2DT(num_filters*8, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2DT(num_filters*4, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2DT(num_filters*2, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    w = conv2DT(3, (3, 3), strides=(2, 2), padding='same', activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=inp, outputs=w, name=name)
    return model


def read_images():
    imagepaths, labels = list(), list()
    dataset_path = 'D:\Storage\Technical\Linux Resources\Images\ArtGen'

    # An ID will be affected to each sub-folders by alphabetical order
    label = 0
    # List the directory
    try:  # Python 2
        classes = sorted(os.walk(dataset_path).next()[1])
    except Exception:  # Python 3
        classes = sorted(os.walk(dataset_path).__next__()[1])
    # List each sub-directory (the classes)
    for c in classes:
        c_dir = os.path.join(dataset_path, c)
        try:  # Python 2
            walk = os.walk(c_dir).next()
        except Exception:  # Python 3
            walk = os.walk(c_dir).__next__()
        # Add each image to the training set
        for sample in walk[2]:
            # Only keeps jpeg images
            #if sample.endswith('.jpg') or sample.endswith('.jpeg'):
            imagepaths.append(os.path.join(c_dir, sample))
            labels.append(label)
        label += 1
    train, test = train_test_split(np.array(imagepaths), train_size=0.8)
    num_batches = int(len(train)/batch_size)
    return train, test


def load_image(image_path):
    try:
        image = tf.keras.preprocessing.image.load_img(image_path)
        input_arr = np.expand_dims(np.array(tf.keras.preprocessing.image.img_to_array(image)) / 255.0, axis=0)
    except Exception:
        try:
            os.remove(image_path)
        except Exception:
            print("Exception: image invalid, unable to delete image.")
        input_arr = np.random.rand(1, 256, 256, 3)
    return input_arr


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


def test_model(model, source, style):
    _, _, new_img = model(source, style)

    plt.figure(figsize=(6, 6))

    plt.subplot(3, 2, 1)
    plt.imshow(source[0])
    plt.grid(False)

    plt.subplot(3, 2, 2)
    plt.imshow(style[0])
    plt.grid(False)

    plt.subplot(3, 2, 3)
    plt.imshow(Image.fromarray(np.array(new_img[0]), 'RGB'))
    plt.grid(False)

    plt.subplot(3, 2, 4)
    indices = [i for i in range(len(model.loss))]
    plt.plot(indices, model.loss)
    plt.grid(False)

    plt.show()


class AE_A(tf.keras.Model):
    def __init__(self):
        super(AE_A, self).__init__()
        self.encoder1 = enc("enc1")
        self.encoder2 = enc("enc2")
        self.decoder = dec("dec")
        self.compile(optimizer=optimizer)
        self.loss = []

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
        encoded = tf.concat([source_encoded, style_encoded], axis=1)
        decoded = self.decoder(encoded)

        return source_encoded, style_encoded, decoded

    def train_model(self, fname=None):
        try:
            self.load(fname)
            print("File load successful.")
        except Exception:
            print("File load failed.")

        image_dataset, dataset_size = create_dataset()
        dataset_size = dataset_size-1
        print("Dataset size is", dataset_size)
        print("Total number of images is", dataset_size*batch_size)

        for i in range(num_epochs):
            print("Starting epoch {}/{}".format(i, num_epochs))
            start = time.time()
            batch_on = 0
            for source, style in zip(image_dataset.take(300), image_dataset.take(300)):
                try:
                    hold = self.train_step(source, style, optimizer)
                    if batch_on % 50 == 0:
                        print("Beginning batch #" + str(batch_on), 'of size', batch_size)
                        self.loss += [hold]
                except Exception:
                    print("Batch #", batch_on, "failed. Continuing with next batch.")
                batch_on += 1
            duration = time.time()-start
            print(int(duration/60), "minutes &", int(duration % 60), "seconds, for epoch", i)
            if i % 5 == 0:
                test_model(self, source, style)
            print('\n')
            image_dataset, _ = create_dataset()
            self.save("test")
            time.sleep(5)

    def train_step(self, source, style, optimizer):
        with tf.GradientTape() as tape:
            encoder2 = self.encoder2
            _, _, prediction = self(source, style)
            _, _, source_reconstruction = self(source, source)
            # _, _, style_reconstruction = self(style, style)
            loss = tf.abs(0.8 * tf.reduce_mean((source-prediction)**2) +
                    0.2 * tf.reduce_mean((encoder2(style)-encoder2(prediction))**2) +
                    reconstruction_learning_rate * tf.reduce_mean((source-source_reconstruction)**2)) * learning_rate
            # print("\nContent error:", tf.reduce_mean((source-prediction)**2))
            # print("Style error:", tf.reduce_mean((encoder2(style)-encoder2(prediction))**2))
            # print("Identity error:", tf.reduce_mean((source-source_reconstruction)**2))

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

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