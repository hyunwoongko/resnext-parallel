"""
@author : Hyunwoong
@when : 2019-10-15
@homepage : https://github.com/gusdnd852
"""
import tensorflow as tf
from keras.layers import Lambda

from models.layers.gconv2d_backend import GroupConv2D_Backend


class GroupConv2D_Parallel:

    def __init__(self, _in, _out, batch_size, kernel_size, strides, cardinality, weight_decay):
        """
        Group Convolution Layer with Parallel Support using tf.while_loop
        """
        self._in = _in // cardinality
        self._out = _out // cardinality
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.strides = strides
        self.cardinality = cardinality
        self.weight_decay = weight_decay

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return Lambda(lambda z: self.__forward(z))(x)

    def __forward(self, x):
        i, outs = tf.constant(0), tf.TensorArray(dtype=tf.float32, size=self.cardinality)
        x, i, outs = tf.while_loop(self.cond, self.body, [self.projection(x), i, outs])
        outs = [outs.gather([i]) for i in range(self.cardinality)]
        out = tf.concat(outs, axis=4)

        _, b, w, h, c = out.shape
        out = tf.reshape(out, (b, w, h, c))
        return out

    def body(self, x, i, outs):
        group = tf.gather(x, i)
        group = GroupConv2D_Backend(filters=self._out,
                                    kernel_size=self.kernel_size,
                                    padding='same',
                                    use_bias=False,
                                    strides=self.strides,
                                    kernel_initializer='he_normal',
                                    kernel_regularizer=self.weight_decay)(group)

        outs = outs.write(i, group)
        i = tf.add(i, 1)  # increase index to proceed loop
        return x, i, outs

    def cond(self, x, i, outs):
        return tf.less(i, self.cardinality)

    def projection(self, x):
        _, w, h, c = x.shape
        out = c // self.cardinality

        x = tf.reshape(x, (self.batch_size, c, w, h))  # 1. change to channel first to prevent wrong reshape
        x = tf.reshape(x, (self.cardinality, self.batch_size, out, w, h))  # 2. do linear projection
        x = tf.reshape(x, (self.cardinality, self.batch_size, w, h, out))  # 3. restore to channel last
        return x
