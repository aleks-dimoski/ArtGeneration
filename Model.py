import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.python.keras.backend import set_session
import functools
from tensorflow.keras.layers import Input, Concatenate
from PIL import Image

#assert len(tf.config.list_physical_devices('GPU')) > 0

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
set_session(sess)

num_epochs = 10
num_filters = 16
batch_size = 16
learning_rate = 2e-2
latent_dim = 30
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


def enc():
    x = Input(shape=(128, 128, 3))

    ### Layer 0 ###
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    w = tfa.layers.InstanceNormalization()(x)
    #x = tf.keras.layers.MaxPool2D((2, 2))(w)
    x = w

    ### Layer 1 ###
    y = encoder()(x)
    w = y#Concatenate()([x, y])
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(w)
    #y = encoder()(x)
    #w = y#Concatenate()([x, y])
    #x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), activation='relu')(w)
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
    y = tf.keras.layers.Flatten()(x)
    z = tf.keras.layers.Dense(32*32*3/2, activation='relu')(y)
    x = z#tf.keras.layers.Dense(1, activation='sigmoid')(z)

    model = net(inputs=Input(shape=(128, 128, 3)), outputs=x, name='Encoder')
    return model


def enc5():
    x = Input(shape=(128, 128, 3))

    ### Layer 0 ###
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(7, 7), activation='relu')(x)
    w = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(w)
    '''
    ### Layer 1 ###
    y = encoder()(x)
    w = y#Concatenate()([x, y])
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), activation='relu')(w)
    y = encoder()(x)
    w = y#Concatenate()([x, y])
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), activation='relu')(w)

    ### Layer 2 ###
    y = encoder()(x)
    w = y#Concatenate()([x, y])
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), strides=2, activation='relu')(w)
    y = encoder()(x)
    w = y#Concatenate()([x, y])
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), activation='relu')(w)
    '''
    ### Classification ###
    y = tf.keras.layers.Flatten()(x)
    z = tf.keras.layers.Dense(512, activation='relu')(y)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(z)

    model = net(inputs=Input(shape=(128, 128, 3)), outputs=x, name='Encoder5')
    return model


def dec():
    conv2DT = tf.keras.layers.Conv2DTranspose
    normalize = tf.keras.layers.BatchNormalization
    inp = Input(shape=(32, 32, 3))

    ### ????? ###
    x = conv2DT(num_filters, (3, 3), strides=(1,1), padding='same', activation='relu')(inp)
    w = normalize()(x)
    x = conv2DT(num_filters, (3, 3), strides=(2,2), padding='same', activation='relu')(w)
    w = normalize()(x)
    x = conv2DT(num_filters, (3, 3), strides=(2,2), padding='same', activation='relu')(w)
    w = normalize()(x)
    x = conv2DT(3, (3, 3), strides=(1,1), padding='same', activation='relu')(w)
    #w = normalize()(x)
    print(x.shape)
    w = tf.keras.layers.Reshape(target_shape=(128, 128, 3))(x)

    model = net(inputs=inp, outputs=w, name='Decoder1')
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

    # Convert to Tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    # Build a TF Queue, shuffle data
    dataset = tf.data.Dataset.from_tensor_slices((imagepaths, labels))

    def _parse_function(filename, label):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32)
        return image, label

    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(batch_size=batch_size)

    # step 4: create iterator and final input tensor
    iterator = dataset.as_numpy_iterator()
    return iterator


def train_step(source, style, model, optimizer):
    enc1 = model.encoder1
    enc2 = model.encoder2
    enc5 = model.encoder5
    deco = model.decoder
    with tf.GradientTape() as tape:
        loss5 = tf.reduce_mean(tf.abs(source), axis=(1, 2, 3))
        lossA = tf.reduce_mean(tf.abs(source - deco(tf.concat(enc1(source), enc2(source))))+
                               tf.abs(source - deco(tf.concat(enc1(style), enc2(style))))+
                               loss5, axis=(1, 2, 3))
    grads = tape.gradient(lossA, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    grads = tape.gradient(loss5, enc5.trainabile_variables)
    optimizer.apply_gradients(zip(grads, enc5.trainable_variables))
    return loss5, lossA


def train_model(model):
    ite = read_images()

    with tf.compat.v1.Session() as ses:
        for i in range(num_epochs):
            print("\nStarting epoch {}/{}".format(i + 1, num_epochs))
            for idx in range(0, ite.__sizeof__(), batch_size):
                # First grab a batch of training data and convert the input images to tensors
                source = ite.next()
                style = ite.next()
                source = tf.convert_to_tensor(source, dtype=tf.float32)
                style = tf.convert_to_tensor(style, dtype=tf.float32)
                # print(tf.shape(images[0]))
                loss = train_step(source, style, model, optimizer)
    model.save()


class net(tf.keras.Model):

    def __init__(self, inputs=None, outputs=None, name=None):
        super(net, self).__init__()
        self.inputs = inputs
        self.outputs = outputs


class AE_A(tf.keras.Model):
    def __init__(self):
        super(AE_A, self).__init__()
        self.encoder1 = enc()
        self.encoder2 = enc()
        self.encoder5 = enc5()
        self.decoder = dec()

    def save(self, fname):
        dir = os.getcwd() + '\\'
        if not os.path.isdir(dir + fname):
            try:
                os.mkdir(dir + fname)
            except OSError:
                print("File save failed.")
        self.encoder1.save(dir + fname + '\encoder1')
        self.encoder2.save(dir + fname + '\encoder2')
        self.encoder5.save(dir + fname + '\encoder5')
        self.decoder.save(dir + fname + '\decoder')

    def load(self, fname):
        dir = os.getcwd() + '/'
        self.encoder1 = tf.keras.models.load_model(dir + fname + '\encoder1')
        self.encoder2 = tf.keras.models.load_model(dir + fname + '\encoder2')
        self.encoder5 = tf.keras.models.load_model(dir + fname + '\encoder5')
        self.decoder = tf.keras.models.load_model(dir + fname + '\decoder')

    def train(self):
        train_model(self)

    def predict(self, source, style):
        y = self.decoder(tf.concat(self.encoder1(source), self.encoder2(style)))
        plt.imshow(np.squeeze(y))
        plt.show


model = AE_A()
model.train()
model.save('test')


