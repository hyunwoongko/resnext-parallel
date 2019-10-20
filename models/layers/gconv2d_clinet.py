"""
@author : Hyunwoong
@when : 2019-10-15
@homepage : https://github.com/gusdnd852
"""
from keras.layers import Conv2D, BatchNormalization, Activation

from models.layers.gconv2d_parallel import GroupConv2D_Parallel


class GConv2D:
    def __init__(self, _in, _out, kernel_size, batch_size, cardinality, strides=1, weight_decay=5e-4):
        self._in = _in
        self._out = _out
        self.cardinality = cardinality
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.strides = strides
        self.weight_decay = weight_decay

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        if self.cardinality == 1:
            return self.__conv_1times(x)
        else:
            return self.__conv_ntimes_parallel(x)

    def __conv_1times(self, x):
        x = Conv2D(self._out,
                   kernel_size=self.kernel_size,
                   padding='same',
                   use_bias=False,
                   strides=self.strides,
                   kernel_initializer='he_normal',
                   kernel_regularizer=self.weight_decay)(x)

        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        return x

    def __conv_ntimes_parallel(self, x):
        x = GroupConv2D_Parallel(_in=self._in,
                                 _out=self._out,
                                 batch_size=self.batch_size,
                                 kernel_size=self.kernel_size,
                                 strides=self.strides,
                                 cardinality=self.cardinality,
                                 weight_decay=self.weight_decay)(x)

        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        return x