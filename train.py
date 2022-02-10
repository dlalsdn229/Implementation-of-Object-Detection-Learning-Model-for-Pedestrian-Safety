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


#from tensorflow.keras.preprocessing.image import ImageDataGenerator

#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


'''
X_train, X_test, y_train, y_test = np.load("test1000.npy",allow_pickle=True)
print(X_train.shape)
print(X_train.shape[0])
print(np.bincount(y_train))
print(np.bincount(y_test))
'''
#data_aug_gen = ImageDataGenerator(rescale=1./255, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.5,zoom_range=[0.8,0.2],horizontal_flip=True,vertical_flip=True, fill_mode='nearest')
#train_generator = data_aug_gen.flow_from_directory('E:/DR detection datasets/trainset/aug/',target_size=(512,512),batch_size=32,class_mode='binary')

h = h5py.File('train_numpy.h5', 'r')

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

    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    model = Sequential()
    model.add(Conv2D(32,(3,3), padding="same", input_shape=X_train.shape[1:], activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (1,1), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))


   # model.add(MaxPooling2D(pool_size=(2,2)))
   # model.add(Flatten())
    #  model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


    model_dir = './model'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_path = model_dir + "/model.tflite"

    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_acc', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', mode=min, patience=7)


model.summary()


history = model.fit(X_train, y_train, batch_size=4, validation_split=0.3, epochs=100, callbacks=[checkpoint, early_stopping])#
#history = model.fit_generator(train_generator,steps_per_epoch=200, epochs=10)
print("정확도 : %.2f " %(model.evaluate(X_test, y_test)[1]))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss', 'acc', 'val_acc'], loc='upper left')
plt.show()

converter = tf.lite.TFLiteConverter.from_keras_model(model_path)
tflite_model = converter.convert()
open("model.tflite","wb").write(tflite_model)

