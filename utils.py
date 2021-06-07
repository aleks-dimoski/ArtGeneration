import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import datetime
from scipy import stats


def create_dataset(batch_size=4):
    dgen_params = dict(
        rescale=1. / 255,
        shear_range=0.06,
        zoom_range=0.06,
        horizontal_flip=True,
        brightness_range=(0.9, 1.1),
    )
    gen_params = dict(
        batch_size=batch_size,
        color_mode='rgb',
        class_mode=None,
        #steps_per_epoch
    )
    image_dgen = tf.keras.preprocessing.image.ImageDataGenerator(**dgen_params)
    image_gen = image_dgen.flow_from_directory('D:\Storage\Technical\Linux Resources\Images\ArtGen', **gen_params)
    image_dataset = tf.data.Dataset.from_generator(lambda: image_gen,
                                                   output_signature=tf.TensorSpec(shape=(None, 256, 256, 3),
                                                                                  dtype=tf.float32))
    return image_dataset, len(image_gen)


def get_size(tsr):
    try:
        tsr = tsr.get_shape().as_list()
    except Exception:
        pass
    prod = 1
    for i in tsr:
        if not i == None:
            prod *= i
    return prod


def print_time_remaining(cur_epoch, tot_epochs, time_taken):
    remaining = tot_epochs - cur_epoch
    print(f'Time taken for epoch {cur_epoch} is {str(datetime.timedelta(seconds=time_taken))}')
    print("Estimated time remaining is", str(datetime.timedelta(seconds=time_taken*remaining)))



def test_model(model, source=None, style=None, num='0', test=False, name='test', details=''):
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
        plt.savefig(os.path.join('pred' + name, f'epoch_{num}_{details}.png'))
        #plt.show()
    else:
        pred.save(os.path.join('pred' + name, 'epoch_' + str(num) + '.png'))


def record_steps(num=0):
    num -= num % 8
    num -= num / 8
    num -= num % 4
    return num / 4


@tf.autograph.experimental.do_not_convert
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon