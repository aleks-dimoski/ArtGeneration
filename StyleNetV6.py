import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
import time
import PIL
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
num_epochs = 50
num_filters = 64
batch_size = 24
learning_rate = 2e-4
record_epochs = 2
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model_name = 'V6'

conv2DT = tf.keras.layers.Conv2DTranspose
conv2D = tf.keras.layers.Conv2D


def enc_unit(inp, filter_mult=1, name='enc', first=False):
    if first:
        x = conv2D(num_filters*filter_mult*2, kernel_size=(7, 7), strides=(2, 2), padding='same')(inp)
        x = tfa.layers.InstanceNormalization()(x)
        x = tf.keras.layers.MaxPool2D()(x)
        return x

    x = conv2D(num_filters * filter_mult, kernel_size=(1, 1), strides=(1, 1), padding='same')(inp)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2D(num_filters * filter_mult, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2D(num_filters * filter_mult, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2D(num_filters * filter_mult, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x1 = conv2D(num_filters * filter_mult, kernel_size=(1, 1), strides=(2, 2), padding='same')(inp)
    x = tf.keras.layers.Add()([x, x1])
    x2 = tf.keras.layers.LeakyReLU()(x)

    x = conv2D(num_filters * filter_mult * 2, kernel_size=(1, 1), strides=(1, 1), padding='same')(x2)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2D(num_filters * filter_mult * 2, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2D(num_filters * filter_mult * 2, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv2D(num_filters * filter_mult * 2, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x2 = conv2D(num_filters * filter_mult * 2, kernel_size=(1, 1), strides=(2, 2), padding='same')(x1)
    x = tf.keras.layers.Add()([x, x2])
    x = tf.keras.layers.LeakyReLU()(x)

    return x


def dec_unit(inp, filter_mult=1, name='dec', last=False):
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(inp)
    x = conv2D(num_filters * filter_mult*2, (3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x1 = conv2DT(num_filters * filter_mult*2, (3, 3), strides=(2, 2), padding='same')(inp)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Add()([x, x1])
    if not last:
        x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x1)
        x = conv2D(num_filters * filter_mult, (3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x1 = conv2DT(num_filters * filter_mult, (3, 3), strides=(2, 2), padding='same')(x1)
        x1 = tf.keras.layers.LeakyReLU()(x1)
        x = tf.keras.layers.Add()([x, x1])
    else:
        x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        x = conv2DT(3, (7, 7), strides=(1, 1), padding='same', activation='sigmoid')(x)

    return x


def latent(inp, name='latent'):
    shape = inp.shape[1:]
    x = tf.keras.layers.Flatten()(inp)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(np.prod(shape))(x)
    x = tf.keras.layers.Reshape(target_shape=shape)(x)
    return x


class AE_A(tf.keras.Model):
    def __init__(self):
        super(AE_A, self).__init__()
        self.compile(optimizer=optimizer)
        self.loss_identity = []
        self.kl_loss = []
        self.loss = {}

        inp1 = tf.keras.Input(shape=(32, 32, 3))
        print(inp1.get_shape())
        val = enc_unit(inp=inp1, filter_mult=2, first=True, name='E1')
        val = enc_unit(inp=val, filter_mult=8, name='E2')
        val = latent(inp=val)
        val = dec_unit(inp=val, filter_mult=4, name='D2')
        val = dec_unit(inp=val, last=True, filter_mult=1, name='D1')
        self.model = tf.keras.Model(inputs=inp1, outputs=val, name='aea')

    def save(self, fname):
        dr = os.getcwd() + '\\'
        if not os.path.isdir(dr + fname):
            try:
                os.mkdir(dr + fname)
            except OSError:
                print("File save failed.")
        tf.keras.models.save_model(model=self.model, filepath=dr + fname + '\\' + self.model.name)

    def load(self, fname):
        dr = os.getcwd() + '\\'
        self.model = tf.keras.models.load_model(dr + fname + '\\' + self.model.name)

    #@tf.function
    def call(self, x):
        return self.model(x)

    def merge(self, source, style, slice=1):
        return self(source)

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
        record_batch = utils.record_steps(int(dataset_size))

        start_time = time.time()
        print("Beginning training at", start_time)

        for i in range(num_epochs):
            print("Starting epoch {}/{}".format(i, num_epochs))
            start = time.time()
            batch_on = 0
            for source in zip(image_dataset.take(int(dataset_size))):
                source = utils.get_random_crop(np.array(source), 32, 32)
                loss_identity = train_step(self, source, optimizer)
                if batch_on % record_batch == 0:
                    print("Beginning batch #" + str(batch_on), 'out of', int(dataset_size), 'of size', batch_size)
                    self.loss_identity += [loss_identity]
                batch_on += 1
            if i % record_epochs == 0:
                self.loss['Identity'] = self.loss_identity
                utils.test_model(self, source, num=i, test=True, name=model_name, details='identity')
                '''for style in zip(image_dataset.take(1)):
                    style = utils.get_random_crop(style, 32, 32)
                    utils.test_model(self, source, style, num=i, test=True, name=model_name, details='transfer')
                    break'''
            print('\n')
            duration = time.time() - start
            utils.print_time_remaining(i, num_epochs, duration)
            image_dataset, _ = utils.create_dataset(batch_size=batch_size)
            self.save(fname)
            time.sleep(.5)
        print('Training completed in', int((time.time() - start_time) / 60), "minutes &", int(duration % 60), "seconds")

    def build_graph(self):
        print(self.model.summary())
        return self.model


@tf.function
def train_step(model, source, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(source)
        loss_identity = identity_lr * learning_rate * (0.2 * tf.reduce_mean(tf.reduce_sum(cross_entropy(source, prediction)))
                                                       + tf.reduce_mean(tf.reduce_sum((source-prediction)**2)))
        loss = loss_identity

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_identity

model = AE_A()
tf.keras.utils.plot_model(model.build_graph(), model_name+".png", show_shapes=True, expand_nested=True)
#model.train_model(model_name)
model.load(model_name)
model = model.model
model.trainable = False
image_dataset, _ = utils.create_dataset(batch_size=1)

for layer in model.layers:
    print(layer.name)

content_layers = ['conv2d_transpose_3']

style_layers = ['conv2d',
                'conv2d_1',
                'conv2d_2',
                'conv2d_3',
                'conv2d_4',
                'conv2d_5',
                'conv2d_6',
                'conv2d_7',
                'conv2d_8',
                'conv2d_9',
                'conv2d_10']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def aea_layers(layer_names, model):
    outputs = [model.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([model.input], outputs)
    return model


style_extractor = aea_layers(style_layers, model)

for source, style in zip(image_dataset.take(1), image_dataset.take(1)):
    #utils.test_model(model, source, test=True, name=model_name)
    #utils.test_model(model, source, style, test=True, name=model_name)
    break

source = np.expand_dims(np.array(style)[0], axis=0)
style = np.expand_dims(utils.get_random_crop(np.array(style)[0], 32, 32), axis=0)

style_outputs = style_extractor(style[0]*255)

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.aea = aea_layers(style_layers + content_layers, model)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.aea.trainable = False

    def call(self, inputs):
        inputs = inputs*255.0
        outputs = self.aea(inputs)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [utils.gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss


def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

    return x_var, y_var


def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))


#@tf.function()
def train_image(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight*tf.image.total_variation(image)
    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def tensor_to_image(tensor, fname=None):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    #plt.imshow(PIL.Image.fromarray(tensor))
    #plt.show()
    if fname:
        PIL.Image.fromarray(tensor).save(fname)


extractor = StyleContentModel(style_layers, content_layers)

style_targets = extractor(style[0])['style']

image = np.array(source)

style_weight = 1e-2
content_weight = 1e4
total_variation_weight = 50

x_deltas, y_deltas = high_pass_x_y(source)

start = time.time()

epochs = 10
steps_per_epoch = 100
step = 0

file_name = 'stylized-image.png'
for x in range(0, int(256/32)-1):
    for y in range(0, int(256/32)-1):
        block = image[:, x*32:(x+1)*32, y*32:(y+1)*32, :]
        results = extractor(tf.constant(block))
        content_targets = extractor(block)['content']
        block = tf.Variable(block)
        for n in range(epochs):
            for m in range(steps_per_epoch):
                train_image(block)
                step += 1
            print(".", end='', flush=True)
        image[:, x * 32:(x + 1) * 32, y * 32:(y + 1) * 32, :] = np.array(block)
        tensor_to_image(image, fname=file_name)
        print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end-start))

