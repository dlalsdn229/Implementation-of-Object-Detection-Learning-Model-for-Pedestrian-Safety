#from PIL import Image
import os, glob, sys, numpy as np
#import keras
import tensorflow as tf
#from keras_applications import vgg16
#from keras import applications
#from keras.utils import np_utils
#from tensorflow.python.keras import regularizers

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as K
import h5py


h = h5py.File('WalkerBollard1.h5', 'r')

X_data = h['img'][:]
y_data = h['label'][:]

X_train, X_test, y_train, y_test = train_test_split(X_data,y_data,test_size=0.3)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


image_w = 128
image_h = 128
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

nb_filters=32
nb_conv=3
nb_pool=2
nb_epoch=5
nb_classes=5


with K.tf_ops.device('/device:GPU:0'):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32,(3,3), padding="same", input_shape=X_train.shape[1:], activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (1, 1), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256,activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation="sigmoid")
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model_dir = './model'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_path = model_dir + "/WBmodel1.model"

    model.fit(X_train, y_train, batch_size=8, validation_split=0.3, epochs=12)

    model.evaluate(X_test,y_test)

