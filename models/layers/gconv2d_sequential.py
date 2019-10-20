"""
@author : Hyunwoong
@when : 2019-10-15
@homepage : https://github.com/gusdnd852

THIS CODE IS NOT SUPPORT PARALLEL.
IT IS JUST FOR COMPARISON. DON'T USE THIS CODE.
"""
from keras import backend as K
from keras.layers import Conv2D, BatchNormalization, Activation, Concatenate, Lambda
from keras.regularizers import l2


class GConv2D:
    def __init__(self, _in, _out, kernel_size, cardinality, strides=1, weight_decay=5e-4):
        """
        constructor of GConv2D layer

        :param _in: input channel
        :param _out: output channel
        :param kernel_size:
        :param cardinality:
        :param strides:
        :param weight_decay:
        """
        self._in = _in
        self._out = _out
        self.cardinality = cardinality
        self.kernel_size = kernel_size
        self.strides = strides
        self.weight_decay = weight_decay

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        forward to group convolution 2D layer

        :param x: input
        :return: output of layers
        """
        x = self.__grouped_convolution_block(x,
                                             _in=self._in,
                                             _out=self._out,
                                             kernel_size=self.kernel_size,
                                             cardinality=self.cardinality,
                                             strides=self.strides,
                                             weight_decay=self.weight_decay)
        return x

    def __grouped_convolution_block(self, x, kernel_size, _in, _out,
                                    cardinality, strides, weight_decay=5e-4):
        """
        splitting group and forwarding convolution layer & batch norm & activation

        :param x: input
        :param kernel_size: size of kernel
        :param _in: number of input channel
        :param _out: number of output channel
        :param cardinality: number of group
        :param strides: strides
        :param weight_decay: factor for weight decay
        :return: output of layers
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        group_list = list()

        if cardinality == 1:
            x = Conv2D(_out,
                       kernel_size=kernel_size,
                       padding='same',
                       use_bias=False,
                       strides=(strides, strides),
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2(weight_decay))(x)

            x = BatchNormalization(axis=channel_axis)(x)
            x = Activation('relu')(x)
            return x

        for c in range(cardinality):
            group = Lambda(lambda z: self.split(z, c, int(_in / cardinality)))(x)
            group = Conv2D(int(_out / cardinality),
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=False,
                           strides=(strides, strides),
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(weight_decay))(group)

            group_list.append(group)

        group_merge = Concatenate(axis=channel_axis)(group_list)
        x = BatchNormalization(axis=channel_axis)(group_merge)
        x = Activation('relu')(x)
        return x

    def split(self, x, c, _in_channel_split):
        """
        splitting a given input into groups

        :param x: input
        :param c: number of group (cardinality)
        :param _in_channel_split: input channel of each group (input_channel / cardinality)
        :return: splitted input
        """
        _from = c * _in_channel_split
        _to = (c + 1) * _in_channel_split
        return x[:, :, :, _from:_to]
