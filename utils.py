import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


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


def test_model(model, source, style, num='0', test=False, name='test'):
    _, _, new_img = model(source, style)
    pred = Image.fromarray(np.array(new_img[0]), 'RGB')

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

        plt.imshow(pred)
        plt.grid(False)

        plt.subplot(3, 1, 3)
        indices = [i for i in range(len(model.loss_content))]
        plt.plot(indices, model.loss_content, label='Content', color='blue')
        plt.plot(indices, model.loss_style, label='Style', color='green')
        plt.plot(indices, model.loss_identity, label='Identity', color='red')
        plt.legend()
        plt.grid(False)

        plt.show()
    else:
        pred.save(os.path.join('pred' + name, 'epoch_' + str(num) + '.png'))