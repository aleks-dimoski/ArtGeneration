import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D, Add, BatchNormalization, LeakyReLU, Reshape, Flatten, Dense
import utils
import tensorflow.keras.backend as K
import tensorflow_datasets as tfds

input_img = Input(shape=(128, 128, 3))
# Encoder
# strides of 2,2 works like max pooling and downsamples the image
y = Conv2D(32, (3, 3), padding='same',strides =(2,2))(input_img)
y = LeakyReLU()(y)
y = Conv2D(64, (3, 3), padding='same',strides =(2,2))(y)
y = LeakyReLU()(y)
y2 = Conv2D(128, (3, 3), padding='same',strides =(2,2))(y)
y = LeakyReLU()(y2)
y = Conv2D(256, (3, 3), padding='same',strides =(2,2))(y)
y = LeakyReLU()(y)
y1 = Conv2D(256, (3, 3), padding='same',strides =(2,2))(y)
y = LeakyReLU()(y1)
y = Conv2D(512, (3, 3), padding='same',strides =(2,2))(y)
y = LeakyReLU()(y)
y = Conv2D(1024, (3, 3), padding='same',strides =(2,2))(y)
y = LeakyReLU()(y)
vol = y.shape # shape of the final convolutional layer
x = Flatten()(y)
latent = Dense(128,activation='relu')(x) # bottleneck layer to control the information flow

def lrelu_bn(inputs):
 lrelu = LeakyReLU()(inputs)
 bn = BatchNormalization()(lrelu)
 return bn

# Decoder
y = Dense(np.prod(vol[1:]), activation='relu')(latent) # accepting the output from the bottleneck layer
y = Reshape((vol[1], vol[2], vol[3]))(y)
y = Conv2DTranspose(1024, (3,3), padding='same')(y)
y = LeakyReLU()(y)
y = Conv2DTranspose(512, (3,3), padding='same',strides=(2,2))(y)
y = LeakyReLU()(y)
y = Conv2DTranspose(256, (3,3), padding='same',strides=(2,2))(y)
y= Add()([y1, y]) # remove to run model without skip connections
y = lrelu_bn(y)  # remove to run model without skip connections
y = Conv2DTranspose(256, (3,3), padding='same',strides=(2,2))(y)
y = LeakyReLU()(y)
y = Conv2DTranspose(128, (3,3), padding='same',strides=(2,2))(y)
y= Add()([y2, y]) # remove to run model without skip connections
y = lrelu_bn(y) # remove to run model without skip connections
y = Conv2DTranspose(64, (3,3), padding='same',strides=(2,2))(y)
y = LeakyReLU()(y)
y = Conv2DTranspose(32, (3,3), padding='same',strides=(2,2))(y)
y = LeakyReLU()(y)
y = Conv2DTranspose(3, (3,3), activation='sigmoid', padding='same',strides=(2,2))(y)

model_1 = tf.keras.Model(input_img,y)
model_1.summary()

def make_iterator(dataset):
    iterator = dataset.make_one_shot_iterator()
    next_val = iterator.get_next()

    with K.get_session().as_default() as sess:
        while True:
            *inputs, labels = sess.run(next_val)
            yield inputs, labels

X_train = utils.create_dataset(32)
X_train = tfds.as_numpy(X_train)

model_1.compile(optimizer=tf.keras.optimizers.Adam(0.001,beta_1=0.9), loss='binary_crossentropy',metrics=['accuracy'])
history_2 = model_1.fit(X_train,batch_size = 32,epochs = 200)
