from keras.layers import Layer, add
import keras.backend as K
import math
import tensorflow as tf


def get_timing_signal_1d(length,
                         channels,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
    position = tf.to_float(tf.range(length) + start_index)
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


class PosEmb(Layer):
    def __init__(self, length, channels, **kwargs):
        super(PosEmb, self).__init__(**kwargs)
        self._len = length
        self._channels = channels

    def call(self, inputs, **kwargs):
        pos_emb = get_timing_signal_1d(self._len, self._channels)
        out = add([inputs, pos_emb])
        return out


class Reverse(Layer):
    def __init__(self, length, channels, **kwargs):
        super(Reverse, self).__init__(**kwargs)
        self._len = length
        self._channels = channels

    def call(self, inputs, **kwargs):
        return K.reshape(K.reverse(inputs, axes=-1), shape=(-1, self._len, self._channels))

    def compute_mask(self, inputs, mask=None):
        return mask


class Mask(Layer):
    def __init__(self, mask_value=0, **kwargs):
        super(Mask, self).__init__(**kwargs)
        self._mask_val = mask_value

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, self._mask_val)
