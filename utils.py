import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import os
import cv2
import datetime
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.io import imread, imshow
from sklearn.cluster import KMeans


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
        target_size=(256, 256)
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


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor) #forms the gram matrix.
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


def test_model(model, source=None, style=None, num='0', test=False, name='test', details=''):
    source = np.array(source)
    if style is None:
        source = np.reshape(source[0], (1, source.shape[1], source.shape[2], source.shape[3]))
        new_img = model.call(source)
        style = np.ones_like(source)
    else:
        style = np.array(style)
        source = np.reshape(source[0], (1, source.shape[1], source.shape[2], source.shape[3]))
        style = np.reshape(style[0], (1, style.shape[1], style.shape[2], style.shape[3]))
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
        plt.close()
    else:
        pred.save(os.path.join('pred' + name, 'epoch_' + str(num) + '.png'))


def record_steps(num=0):
    num -= num % 8
    num -= num / 8
    num -= num % 4
    return num / 4


def image_to_pandas(image):
    df = pd.DataFrame([image[:, :, 0].flatten(),
                       image[:, :, 1].flatten(),
                       image[:, :, 2].flatten()]).T
    df.columns = ['Red_Channel', 'Green_Channel', 'Blue_Channel']
    return df


def image_segmentation(img, img2=None, n=4):
    img = np.reshape(np.array(img[0]), (256, 256, 3))
    plt.figure(figsize=(4, 4))
    plt.subplot(2, 2, 1)
    imshow(img)
    df = image_to_pandas(img)
    plt.subplot(2, 2, 2)
    kmeans = KMeans(n_clusters=n, random_state=42).fit(df)
    result = kmeans.labels_.reshape(img.shape[0], img.shape[1])
    imshow(result, cmap='gray')
    if not img2 is None:
        plt.subplot(2, 2, 3)
        img2 = np.reshape(np.array(img2[0]), (256, 256, 3))
        imshow(img2)
        df = image_to_pandas(img2)
        plt.subplot(2, 2, 4)
        kmeans = KMeans(n_clusters=n, random_state=42).fit(df)
        result = kmeans.labels_.reshape(img2.shape[0], img2.shape[1])
        imshow(result, cmap='gray')
    plt.show()
    pixel_plotter(df)
    pixel_plotter_clusters(df, result)


def pixel_plotter(df):
    x_3d = df['Red_Channel']
    y_3d = df['Green_Channel']
    z_3d = df['Blue_Channel']

    color_list = list(zip(df['Red_Channel'].to_list(),
                          df['Blue_Channel'].to_list(),
                          df['Green_Channel'].to_list()))
    norm = colors.Normalize(vmin=0, vmax=1.)
    norm.autoscale(color_list)
    p_color = norm(color_list).tolist()

    fig = plt.figure(figsize=(12, 10))
    ax_3d = plt.axes(projection='3d')
    ax_3d.scatter3D(xs=x_3d, ys=y_3d, zs=z_3d,
                    c=p_color, alpha=0.55);

    ax_3d.set_xlim3d(0, x_3d.max())
    ax_3d.set_ylim3d(0, y_3d.max())
    ax_3d.set_zlim3d(0, z_3d.max())
    ax_3d.invert_zaxis()

    ax_3d.view_init(-165, 60)
    plt.show()


def pixel_plotter_clusters(df, result):
    df['cluster'] = result.flatten()
    x_3d = df['Red_Channel']
    y_3d = df['Green_Channel']
    z_3d = df['Blue_Channel']

    fig = plt.figure(figsize=(12, 10))
    ax_3d = plt.axes(projection='3d')
    ax_3d.scatter3D(xs=x_3d, ys=y_3d, zs=z_3d,
                    c=df['cluster'], alpha=0.55);

    ax_3d.set_xlim3d(0, x_3d.max())
    ax_3d.set_ylim3d(0, y_3d.max())
    ax_3d.set_zlim3d(0, z_3d.max())
    ax_3d.invert_zaxis()

    ax_3d.view_init(-165, 60)
    plt.show()


def get_random_crop(image, crop_height, crop_width):
    image = image[0]
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height
    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)
    crop = image[y: y + crop_height, x: x + crop_width]
    return crop


@tf.autograph.experimental.do_not_convert
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def test_k_means():
    image_dataset, dataset_size = create_dataset(batch_size=1)
    for img, img2 in zip(image_dataset.take(1), image_dataset.take(1)):
        for i in range(2, 6):
            image_segmentation(img, img2=img2, n=i)
        break

#test_k_means()