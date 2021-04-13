import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.python.keras.backend import set_session
import functools
from tensorflow.keras.layers import Input, Concatenate
from PIL import Image
from sklearn.model_selection import train_test_split

#assert len(tf.config.list_physical_devices('GPU')) > 0

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.disable_eager_execution()
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
    inp = Input(shape=(256, 256, 3))

    ### Layer 0 ###
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inp)
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
    z = tf.keras.layers.Dense(64*64*3/2, activation='relu')(y)
    x = z#tf.keras.layers.Dense(1, activation='sigmoid')(z)

    model = tf.keras.models.Model(inputs=inp, outputs=x, name='Encoder')
    return model


def enc5():
    inp = Input(shape=(256, 256, 3))

    ### Layer 0 ###
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), padding='same', activation='relu')(inp)
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
    z = tf.keras.layers.Dense(64, activation='relu')(y)
    #x = tf.keras.layers.Dense(1, activation='sigmoid')(z)

    model = tf.keras.models.Model(inputs=inp, outputs=z, name='Encoder5')
    return model


def dec():
    conv2DT = tf.keras.layers.Conv2DTranspose
    normalize = tf.keras.layers.BatchNormalization
    inp = Input(shape=(64, 64, 3))

    ### ????? ###
    x = conv2DT(num_filters, (3, 3), strides=(1,1), padding='same', activation='relu')(inp)
    w = normalize()(x)
    x = conv2DT(num_filters, (3, 3), strides=(2,2), padding='same', activation='relu')(w)
    w = normalize()(x)
    x = conv2DT(num_filters, (3, 3), strides=(2,2), padding='same', activation='relu')(w)
    w = normalize()(x)
    x = conv2DT(3, (3, 3), strides=(1,1), padding='same', activation='relu')(w)
    #w = normalize()(x)
    w = tf.keras.layers.Reshape(target_shape=(256, 256, 3))(x)

    model = tf.keras.Model(inputs=inp, outputs=w, name='Decoder1')
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
    train, test = train_test_split(np.array(imagepaths), train_size=1)
    return np.array_split(train, batch_size)


def load_image(fnames):
    images = np.expand_dims(np.empty(shape=(256, 256, 3)), axis=0)
    for image_path in fnames:
        image = tf.keras.preprocessing.image.load_img(image_path)
        input_arr = np.expand_dims(np.array(tf.keras.preprocessing.image.img_to_array(image)),axis=0)
        np.append(arr=images, values=np.expand_dims(input_arr, axis=0)) # Convert single images to a batch.
        print(images.shape)
    return images


def train_step(source, style, model, optimizer):
    enc1 = model.encoder1
    enc2 = model.encoder2
    enc5 = model.encoder5
    deco = model.decoder
    with tf.GradientTape() as tape:
        print(enc5(model(source, style)).shape)
        loss5 = tf.reduce_mean(tf.abs(enc5(model(source, style)) - enc5(source)), axis=(1, 2, 3))
        lossA = tf.reduce_mean(tf.abs(source - model(source, source))+
                               tf.abs(source - model(style, style))+
                               loss5, axis=(1, 2, 3))

    grads = tape.gradient(lossA, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    grads = tape.gradient(loss5, enc5.trainabile_variables)
    optimizer.apply_gradients(zip(grads, enc5.trainable_variables))
    return loss5, lossA


def train_model(model):
    dataset = read_images()

    for i in range(num_epochs):
        print("\nStarting epoch {}/{}".format(i + 1, num_epochs))
        for idx in range(0, len(dataset), 2):
            # First grab a batch of training data and convert the input images to tensors
            source = load_image(dataset[idx])
            style = load_image(dataset[idx+1])
            source = tf.convert_to_tensor(source, dtype=tf.float32)
            style = tf.convert_to_tensor(style, dtype=tf.float32)
            # print(tf.shape(images[0]))
            train_step(source, style, model, optimizer)
    model.save()


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

    def call(self, source, style):
        return self.decoder(tf.concat(self.encoder1(source), self.encoder2(style)))

    def train(self):
        train_model(self)

    def predict(self, source, style):
        y = self.decoder(tf.concat(self.encoder1(source), self.encoder2(style)))
        plt.imshow(np.squeeze(y))
        plt.show

model = AE_A()
model.train()
model.save('test')


