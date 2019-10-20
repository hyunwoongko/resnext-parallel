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
        x, i, outs = tf.while_loop(self.cond, self.body, [self.split(x), i, outs],
                                   parallel_iterations=self.cardinality)
        outs = [outs.gather([i]) for i in range(self.cardinality)]
        out = tf.concat(outs, axis=4)

        _, b, w, h, c = out.shape
        out = tf.reshape(out, (b, w, h, c))
        return out

    def body(self, x, i, outs):
        group = x.gather([i])
        _, _, w, h, c = group.shape
        group = tf.reshape(group, (self.batch_size, w, h, c))
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

    def split(self, x):
        _, w, h, c = x.shape
        out = c // self.cardinality
        arr = tf.TensorArray(size=self.cardinality, dtype=tf.float32)

        for i in range(self.cardinality):
            _from = i * out
            _to = (i + 1) * out
            val = x[:, :, :, _from:_to]
            arr = arr.write(value=val, index=i)

        return arr
