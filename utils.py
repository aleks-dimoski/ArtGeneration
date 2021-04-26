import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from scipy import stats


def create_dataset(batch_size=4):
    dgen_params = dict(
        rescale=1. / 255,
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
    image_gen = image_dgen.flow_from_directory('D:\Storage\Technical\Linux Resources\Images\ArtGen', **gen_params)
    image_dataset = tf.data.Dataset.from_generator(lambda: image_gen,
                                                   output_signature=tf.TensorSpec(shape=(None, 256, 256, 3),
                                                                                  dtype=tf.float32))
    return image_dataset, len(image_gen)


def test_model(model, source=None, style=None, num='0', test=False, name='test'):
    if style == None:
        source = np.reshape(np.array(source[0][0]), (1, 256, 256, 3))
        new_img = model(source)
        style = np.ones_like(source)
    else:
        source = np.reshape(np.array(source[0][0]), (1, 256, 256, 3))
        style = np.reshape(np.array(style[0][0]), (1, 256, 256, 3))
        new_img = model.merge(source, style)

    new_img = np.array(new_img) * 255.
    new_img = new_img.astype(np.uint8)
    pred = Image.fromarray(np.array(new_img[0]), 'RGB')

    if not os.path.isdir(os.path.join('pred' + name)):
        os.mkdir(os.path.join('pred' + name))

    if test:
        pred.save(os.path.join('pred' + name, 'test.png'))

        plt.figure(figsize=(6, 6))

        plt.subplot(3, 2, 1)
        plt.imshow(source[0])
        plt.grid(False)

        plt.subplot(3, 2, 2)
        plt.imshow(style[0])
        plt.grid(False)

        plt.subplot(3, 1, 2)

        plt.imshow(np.array(new_img[0]))
        plt.grid(False)

        try:
            plt.subplot(3, 1, 3)
            indices = [i for i in range(len(model.loss[next(iter(model.loss.keys()))]))]
            for label in model.loss.keys():
                plt.plot(indices, model.loss[label], label=label)
            plt.legend()
            plt.grid(False)
        except Exception:
            plt.grid(False)

        plt.show()
    else:
        pred.save(os.path.join('pred' + name, 'epoch_' + str(num) + '.png'))


def record_steps(num=0):
    num -= num % 4
    return num/4


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
