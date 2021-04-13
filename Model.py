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
#tf.compat.v1.disable_eager_execution()
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


def get_model():
    # Create a simple model.
    inputs = tf.keras.Input(shape=(256, 256, 3))
    outputs = tf.keras.layers.Flatten()(inputs)
    outputs = tf.keras.layers.Dense(4)(outputs)
    outputs = tf.keras.layers.Dense(4)(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def enc(name):
    conv2D = tf.keras.layers.Conv2D
    normalize = tfa.layers.InstanceNormalization
    inp = tf.keras.Input(shape=(256, 256, 3))

    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inp)
    x = tfa.layers.InstanceNormalization()(x)
    #x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = conv2D(num_filters, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = normalize()(x)
    x = conv2D(num_filters, (3, 3), strides=2, padding='same', activation='relu')(x)
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
    w = tf.keras.layers.Dense(64*64*3/2, activation='relu')(x)
    #x = z#tf.keras.layers.Dense(1, activation='sigmoid')(z)

    model = tf.keras.models.Model(inputs=inp, outputs=w, name=name)
    return model


def enc5():
    inp = tf.keras.Input(shape=(256, 256, 3))

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
    z = tf.keras.layers.Dense(512, activation='relu')(y)
    #x = tf.keras.layers.Dense(1, activation='sigmoid')(z)

    model = tf.keras.models.Model(inputs=inp, outputs=z, name="enc5")
    return model

encode5 = enc5()
encode5.compile(optimizer=optimizer, loss="mean_squared_error")


def dec(name):
    conv2DT = tf.keras.layers.Conv2DTranspose
    normalize = tf.keras.layers.BatchNormalization
    inp = tf.keras.Input(shape=(64*64*3))
    w = tf.reshape(inp, shape=(1, 64, 64, 3))

    ### ????? ###
    x = conv2DT(num_filters, (3, 3), strides=(1, 1), padding='same', activation='relu')(w)
    w = normalize()(x)
    x = conv2DT(num_filters, (3, 3), strides=(2, 2), padding='same', activation='relu')(w)
    w = normalize()(x)
    x = conv2DT(num_filters, (3, 3), strides=(2, 2), padding='same', activation='relu')(w)
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
    train, test = train_test_split(np.array(imagepaths), train_size=1)
    print(train.shape)
    return np.array_split(train, batch_size)


def load_image(fnames):
    images = np.expand_dims(np.empty(shape=(256, 256, 3)), axis=0)
    for image_path in fnames:
        image = tf.keras.preprocessing.image.load_img(image_path)
        input_arr = np.expand_dims(np.array(tf.keras.preprocessing.image.img_to_array(image)),axis=0)
        np.append(arr=images, values=np.expand_dims(input_arr, axis=0)) # Convert single images to a batch.
    return images


class AE_A(tf.keras.Model):
    def __init__(self):
        super(AE_A, self).__init__()
        self.encoder1 = enc("enc1")
        self.encoder2 = enc("enc2")
        self.decoder = dec("dec")
        self.compile(
            optimizer=optimizer,
            loss={
                "enc1": tf.keras.losses.MeanSquaredError,
                "enc2": tf.keras.losses.MeanSquaredError,
                "dec": tf.keras.losses.MeanSquaredError,
            },
            loss_weights=[0.2, 0.2, 0.2],
        )

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
        return self.decoder(tf.concat([self.encoder1(source), self.encoder2(style)], axis=1))

    def train_model(self, encoder5):
        dataset = read_images()
        print(len(dataset))

        for i in range(1):
            print("\nStarting epoch {}/{}".format(i + 1, num_epochs))
            for idx in range(0, len(dataset), 2):
                # First grab a batch of training data and convert the input images to tensors
                source = load_image(dataset[idx])
                style = load_image(dataset[idx + 1])
                source = tf.convert_to_tensor(source, dtype=tf.float32)
                style = tf.convert_to_tensor(style, dtype=tf.float32)
                self.train_step(source, style, encoder5, optimizer)
                if idx % 100 == 0:
                    print(idx)
        self.save('test')
        encoder5.save('encoder5')

    def train_step(self, source, style, encoder5, optimizer):
        with tf.GradientTape() as tape:
            loss5 = tf.reduce_mean(tf.abs(encoder5(model(source, style)) - encoder5(source)), axis=1)
        grads = tape.gradient(loss5, encoder5.trainable_variables)
        optimizer.apply_gradients(zip(grads, encoder5.trainable_variables))

        with tf.GradientTape() as tape:
            loss_a = tf.reduce_mean(tf.abs(source - model(source, source)) +
                                    tf.abs(source - model(style, style)) +
                                    loss5, axis=(1, 2, 3))
        grads = tape.gradient(loss_a, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss5, loss_a

    def predict(self, source, style):
        return self.decoder(tf.concat([self.encoder1(source), self.encoder2(style)], axis=1))


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
inp1 = Input(shape=(256, 256, 3))
inp2 = Input(shape=(256, 256, 3))
encode1 = enc("enc1")
encode2 = enc("enc2")
out1 = tf.concat([encode1.outputs,encode2.outputs], axis=1)
decode = dec("dec")
out2 = decode(out1)
disp = tf.keras.Model(
    inputs=[encode1.inputs, encode2.inputs],
    outputs=[out2],
)
tf.keras.utils.plot_model(disp, "test.png", show_shapes=True, expand_nested=True)


model = AE_A()
#model.train_model(encode5)
model.load('test')
model.compile(
            optimizer=optimizer,
            loss={
                "enc1": tf.keras.losses.MeanSquaredError,
                "enc2": tf.keras.losses.MeanSquaredError,
                "dec": tf.keras.losses.MeanSquaredError,
            },
            loss_weights=[0.2, 0.2, 0.2],
        )
images = read_images()
print(images[0][0])
source = tf.keras.preprocessing.image.load_img(images[0][0])
style = tf.keras.preprocessing.image.load_img(images[0][0])
new_img = model(np.expand_dims(np.array(tf.keras.preprocessing.image.img_to_array(source)), axis=0),
                np.expand_dims(np.array(tf.keras.preprocessing.image.img_to_array(style)), axis=0))

plt.figure(figsize=(6, 6))

plt.subplot(3, 2, 1)
plt.imshow(source)
plt.grid(False)

plt.subplot(3, 2, 2)
plt.imshow(style)
plt.grid(False)

plt.subplot(2, 1, 2)
plt.imshow(Image.fromarray(np.squeeze(new_img.numpy(), axis=0), 'RGB'))
plt.grid(False)

plt.show()
