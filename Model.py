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
num_filters = 2
batch_size = 2
learning_rate = 5e-2
optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)


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

    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inp)
    x = tfa.layers.InstanceNormalization()(x)
    #x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = conv2D(num_filters*16, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = normalize()(x)
    x = conv2D(num_filters*8, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = normalize()(x)
    x = conv2D(num_filters*4, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = normalize()(x)
    x = conv2D(num_filters*2, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = normalize()(x)
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    '''
    ### Layer 2 ###
    y = encoder()(x)
    w = y#Concatenate()([x, y])
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), strides=2, activation='relu')(w)
    y = encoder()(x)
    w = y#Concatenate()([x, y])
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), activation='relu')(w)
    '''
    ### Latent Space ###
    x = tf.keras.layers.Flatten()(x)
    w = tf.keras.layers.Dense(16*16*3/2, activation='relu')(x)
    #x = z#tf.keras.layers.Dense(1, activation='sigmoid')(z)

    model = tf.keras.models.Model(inputs=inp, outputs=w, name=name)
    return model


def dec(name):
    conv2DT = tf.keras.layers.Conv2DTranspose
    normalize = tf.keras.layers.BatchNormalization
    inp = tf.keras.Input(shape=(16*16*3))
    w = tf.reshape(inp, shape=(1, 16, 16, 3))

    ### ????? ###
    x = conv2DT(num_filters, (3, 3), strides=(1, 1), padding='same', activation='relu')(w)
    w = normalize()(x)
    x = conv2DT(num_filters*2, (3, 3), strides=(2, 2), padding='same', activation='relu')(w)
    w = normalize()(x)
    x = conv2DT(num_filters*4, (3, 3), strides=(2, 2), padding='same', activation='relu')(w)
    w = normalize()(x)
    x = conv2DT(num_filters*8, (3, 3), strides=(2, 2), padding='same', activation='relu')(w)
    w = normalize()(x)
    x = conv2DT(num_filters*16, (3, 3), strides=(2, 2), padding='same', activation='relu')(w)
    w = normalize()(x)
    x = conv2DT(3, (3, 3), strides=(1, 1), padding='same', activation='relu')(w)
    #w = normalize()(x)
    w = tf.keras.layers.Reshape(target_shape=(256, 256, 3))(x)

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
'''tf.keras.preprocessing.image_dataset_from_directory(
    'D:\Storage\Technical\Linux Resources\Images\ArtGen',
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=0.9,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
)'''


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


def create_dataset(image_paths):
    dgen_params = dict(
        rescale= 1. / 255,
        shear_range=0.0,
        zoom_range=0.0,
        horizontal_flip=True,
        brightness_range=(0.7, 1.2),
    )
    gen_params = dict(
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='categorical',
    )
    image_dgen = tf.keras.preprocessing.image.ImageDataGenerator(**dgen_params)
    image_gen = image_dgen.flow_from_directory('D:\Storage\Technical\Linux Resources\Images\ArtGen',**gen_params)
    image_dataset = tf.data.Dataset.from_generator(lambda: image_gen, output_signature=tf.TensorSpec(shape=(256,256,3), dtype=tf.float32))
    print(image_dataset)
    return image_dataset


def test_model(model, source, style):
    _, _, new_img = model(np.expand_dims(np.array(tf.keras.preprocessing.image.img_to_array(source) / 255.0), axis=0),
                          np.expand_dims(np.array(tf.keras.preprocessing.image.img_to_array(style) / 255.0), axis=0))

    plt.figure(figsize=(6, 6))

    plt.subplot(3, 2, 1)
    plt.imshow(source)
    plt.grid(False)

    plt.subplot(3, 2, 2)
    plt.imshow(style)
    plt.grid(False)

    plt.subplot(3, 2, 3)
    plt.imshow(Image.fromarray(np.squeeze(np.array(new_img * 127.5 + 127.5), axis=0), 'RGB'))
    plt.grid(False)

    plt.subplot(3, 2, 4)
    indices = [i for i in range(len(model.loss))]
    plt.plot(indices, model.loss)
    plt.grid(False)

    plt.show()


'''class DataSetCreator(object):
    def __init__(self, batch_size, image_height, image_width, dataset):
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.dataset = dataset

    def _get_class(self, path):
        pat_splited = tf.strings.split(path, os.path.sep)
        return pat_splited[-2] == CLASS_NAMES

    def _load_image(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return tf.image.resize(image, [self.image_height, self.image_width])

    def _load_labeled_data(self, path):
        label = self._get_class(path)
        image = self._load_image(path)
        return image, label

    def load_process(self, shuffle_size=1000):
        self.loaded_dataset = self.dataset.map(self._load_labeled_data,
                                               num_parallel_calls=tf.data.experimental.AUTOTUNE)

        self.loaded_dataset = self.loaded_dataset.cache()

        # Shuffle data and create batches
        self.loaded_dataset = self.loaded_dataset.shuffle(buffer_size=shuffle_size)
        self.loaded_dataset = self.loaded_dataset.repeat()
        self.loaded_dataset = self.loaded_dataset.batch(self.batch_size)

        # Make dataset fetch batches in the background during the training of the model.
        self.loaded_dataset = self.loaded_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def get_batch(self):
        return next(iter(self.loaded_dataset))
'''

class AE_A(tf.keras.Model):
    def __init__(self):
        super(AE_A, self).__init__()
        self.encoder1 = enc("enc1")
        self.encoder2 = enc("enc2")
        self.decoder = dec("dec")
        self.compile(optimizer=optimizer)
        self.loss=[]

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

        #image_paths = read_images()
        image_dataset = create_dataset('')

        for i in range(num_epochs):

            print("\nStarting epoch {}/{}".format(i + 1, num_epochs))
            start = time.time()
            print(image_dataset.take(50))
            for source, style in zip(image_dataset.take(10), image_dataset.take(10)):
                print("hello")
                hold = self.train_step(np.array(source), np.array(style), optimizer)
                test_model(self, np.array(source), np.array(style))
            self.loss+=[hold]


            '''for idx in range(0, len(image_paths), 2):
                # First grab a batch of training data and convert the input images to tensors
                for image in range(0, len(image_paths[idx])):
                    try:
                        source = load_image(image_paths[idx][image])
                        style = load_image(image_paths[idx+1][image])
                        source = tf.convert_to_tensor(source, dtype=tf.float32)
                        style = tf.convert_to_tensor(style, dtype=tf.float32)
                        hold = self.train_step(source, style, optimizer)
                    except Exception:
                        break
                if idx % 100 == 0:
                    print(idx)
                    self.loss+=[hold]'''

            duration = time.time()-start
            print(int(duration/60), "minutes &", int(duration % 60), "seconds, for epoch", i+1)
            if(i % 5 == 0):
                test_model(self, source, style)
            self.save("test")

    def train_step(self, source, style, optimizer):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        source = np.array(source)/255.0
        style = np.array(source)/255.0

        with tf.GradientTape() as tape:
            encoder2 = self.encoder2
            _, _, prediction = self(source, style)
            _, _, source_reconstruction = self(source, source)
            _, _, style_reconstruction = self(style, style)
            loss = (0.2 * cross_entropy(source, prediction) +
                    0.8 * cross_entropy(encoder2(style), encoder2(prediction)) +
                    reconstruction_learning_rate * cross_entropy(source, source_reconstruction)) * learning_rate

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    def build_graph(self):
        source = Input(shape=(256, 256, 3))
        style = Input(shape=(256, 256, 3))
        return tf.keras.Model(inputs=[source, style], outputs=self.call(source, style))



'''model = enc()
model.compile(optimizer="adamax", loss="mean_squared_error")

# Train the model.
test_input = np.random.random((1, 256, 256, 3))
test_target = np.random.random((1, 6144))
model.fit(test_input, test_target)

# Calling `save('my_model')` creates a SavedModel folder `my_model`.
model.save("my_model")

# It can be used to reconstruct the model identically.
reconstructed_model = tf.keras.models.load_model("my_model")

# Let's check:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)

# The reconstructed model is already compiled and has retained the optimizer
# state, so training can resume:
reconstructed_model.fit(test_input, test_target)
'''

model = AE_A()
tf.keras.utils.plot_model(model.build_graph(), "test.png", show_shapes=True, expand_nested=True)
model.train_model('test')
#model.load('test')
images = read_images()
print(images[0][0])
source = tf.keras.preprocessing.image.load_img(images[0][0])
style = tf.keras.preprocessing.image.load_img(images[1][0])

test_model(model, source, style)