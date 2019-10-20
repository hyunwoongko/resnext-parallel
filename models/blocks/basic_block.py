"""
@author : Hyunwoong
@when : 2019-10-15
@homepage : https://github.com/gusdnd852
"""
from keras import backend as K
from keras import layers
from keras.layers import Activation, BatchNormalization
from keras.regularizers import l2

from models.layers.gconv2d_clinet import GConv2D


class BasicBlock:

    def __init__(self, _in, _out, strides, batch_size, cardinality, weight_decay=5e-4):
        self.channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        self.conv1 = GConv2D(
            _in=_in,
            _out=_out,
            kernel_size=(3, 3),
            cardinality=cardinality,
            batch_size=batch_size,
            weight_decay=l2(weight_decay),
            strides=(strides, strides))

        self.conv2 = GConv2D(
            _in=_out,
            _out=_out,
            kernel_size=(3, 3),
            cardinality=cardinality,
            batch_size=batch_size,
            weight_decay=l2(weight_decay),
            strides=(1, 1))

        self.bn1 = BatchNormalization(axis=self.channel_axis)
        self.bn2 = BatchNormalization(axis=self.channel_axis)
        self.relu = Activation('relu')

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        shortcut = x
        # store shortcut

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if x.shape == shortcut.shape:
            # check shape for addition
            x = layers.add([x, shortcut])

        return self.relu(x)
