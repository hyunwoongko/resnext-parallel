# -*- coding: utf-8 -*-
from __future__ import print_function

from models.resnext import ResNeXt

"""
Created on Wed Oct 16 10:31:51 2019

@author: xingshuli, hyunwoong
"""
import os

import keras
from keras.datasets import cifar10
# from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from util.conf import *

# set GPU config
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # '1' or '0' GPU
home_dir = home_dir_linux
num_classes = 10
img_height, img_width = 32, 32

if K.image_dim_ordering() == 'th':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# the data shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
nb_train_samples = x_train.shape[0]
nb_validation_samples = x_test.shape[0]

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# load model
model = ResNeXt(input_shape=input_shape,
                n_class=num_classes,
                weight_decay=weight_decay,
                batch_size=batch_size,
                cardinality=cardinality).model()

optimizer = SGD(lr=init_lr, momentum=momentum, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# model.summary()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rotation_range=30,
                                   zoom_range=0.1,
                                   horizontal_flip=True)
train_datagen.fit(x_train)
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)

# set learning rate schedule
lr_reduce = ReduceLROnPlateau(monitor=monitor,
                              factor=factor,
                              patience=patience,
                              mode=mode,
                              min_lr=min_lr)
# set callbacks for model fit
callbacks = [lr_reduce]

# model fit
hist = model.fit_generator(train_generator,
                           steps_per_epoch=nb_train_samples // batch_size,
                           epochs=epochs,
                           validation_data=(x_test, y_test),
                           callbacks=callbacks)

# print acc and stored into acc.txt
f = open(home_dir + 'train_acc_cifar10.txt', 'w')
f.write(str(hist.history['acc']))
f.close()
# print val_acc and stored into val_acc.txt
f = open(home_dir + 'val_acc_cifar10.txt', 'w')
f.write(str(hist.history['val_acc']))
f.close()

# print train_loss and stored into train_loss.txt
f = open(home_dir + 'train_loss_cifar10.txt', 'w')
f.write(str(hist.history['loss']))
f.close()

# print val_loss and stored into val_loss.txt
f = open(home_dir + 'val_loss_cifar10.txt', 'w')
f.write(str(hist.history['val_loss']))
f.close()

# save model
save_dir = os.path.join(os.getcwd(), 'Cifar_10_model')
model_name = 'keras_cifar10_model.h5'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, model_name)
model.save(save_path)
print('the model has been saved at %s' % save_path)
