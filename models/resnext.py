"""
@author : Hyunwoong
@when : 2019-10-15
@homepage : https://github.com/gusdnd852
"""

from keras import backend as K, Input, Model
from keras.layers import Conv2D, BatchNormalization, Activation, Dense, GlobalAveragePooling2D
from keras.regularizers import l2

from models.blocks.basic_block import BasicBlock


class ResNeXt:

    def __init__(self, input_shape, n_class, batch_size, cardinality, weight_decay=5e-4):
        """
        ResNeXt-18
        """
        self.channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        self.input_shape = input_shape
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        self.conv1 = Conv2D(filters=64,
                            kernel_size=(3, 3),
                            padding='valid',
                            strides=(1, 1),
                            use_bias=False,
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(weight_decay))

        self.bn1 = BatchNormalization(axis=self.channel_axis)
        self.relu = Activation('relu')
        self.l1_1, self.l1_2 = self.make_layers(BasicBlock, BasicBlock, 64, 64, strides=1, cardinality=cardinality)
        self.l2_1, self.l2_2 = self.make_layers(BasicBlock, BasicBlock, 64, 128, strides=2, cardinality=cardinality)
        self.l3_1, self.l3_2 = self.make_layers(BasicBlock, BasicBlock, 128, 256, strides=2, cardinality=cardinality)
        self.l4_1, self.l4_2 = self.make_layers(BasicBlock, BasicBlock, 256, 512, strides=2, cardinality=cardinality)

        self.gap = GlobalAveragePooling2D()
        self.fc = Dense(n_class, activation='softmax', kernel_initializer='he_normal', use_bias=False)

    def __call__(self, x):
        return self.forward(x)

    def make_layers(self, b1, b2, _in, _out, strides, cardinality):
        return b1(_in, _out,
                  strides=strides, weight_decay=self.weight_decay, batch_size=self.batch_size, cardinality=cardinality), \
               b2(_in, _out,
                  strides=strides, weight_decay=self.weight_decay, batch_size=self.batch_size, cardinality=cardinality)

    def forward(self, x):
        # stem forwarding
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # residual layers
        x = self.l1_2(self.l1_1(x))
        x = self.l2_2(self.l2_1(x))
        x = self.l3_2(self.l3_1(x))
        x = self.l4_2(self.l4_1(x))

        # classification layers
        x = self.gap(x)
        x = self.fc(x)
        return x

    def model(self):
        inputs = Input(shape=self.input_shape)
        outputs = self.forward(inputs)
        model = Model(input=inputs, output=outputs, name='ResNeXt')
        return model
