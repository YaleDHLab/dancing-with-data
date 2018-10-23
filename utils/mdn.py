# adapted from https://raw.githubusercontent.com/omimo/Keras-MDN/master/kmdn/mdn.py
from keras import backend as K
from keras.layers import Dense, Input, merge
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
import math
import keras

class MDN(Layer):
  def __init__(self, output_dim, num_mixes, kernel='unigaussian', **kwargs):
    self.output_dim = output_dim
    self.kernel = kernel
    self.num_mixes = num_mixes

    with tf.name_scope('MDN'):
      self.mdn_mus     = Dense(self.num_mixes * self.output_dim, name='mdn_mus')
      self.mdn_sigmas  = Dense(self.num_mixes, activation=K.exp, name='mdn_sigmas')
      self.mdn_pi      = Dense(self.num_mixes, activation=K.softmax, name='mdn_pi')
    super(MDN, self).__init__(**kwargs)

  def build(self, input_shape):
    self.mdn_mus.build(input_shape)
    self.mdn_sigmas.build(input_shape)
    self.mdn_pi.build(input_shape)
    self.trainable_weights = self.mdn_mus.trainable_weights + \
      self.mdn_sigmas.trainable_weights + \
      self.mdn_pi.trainable_weights
    self.non_trainable_weights = self.mdn_mus.non_trainable_weights + \
      self.mdn_sigmas.non_trainable_weights +
      self.mdn_pi.non_trainable_weights
    self.built = True

  def call(self, x, mask=None):
    with tf.name_scope('MDN'):
      mdn_out = keras.layers.concatenate([
        self.mdn_mus(x),
        self.mdn_sigmas(x),
        self.mdn_pi(x)
      ], name='mdn_outputs')
    return mdn_out

  def get_output_shape_for(self, input_shape):
    return (input_shape[0], self.output_dim)

  def get_config(self):
    config = {
      'output_dim': self.output_dim,
      'num_mixes': self.num_mixes,
      'kernel': self.kernel
    }
    base_config = super(MDN, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_loss_func(self):
    def unigaussian_loss(y_true, y_pred):
      mix = tf.range(start = 0, limit = self.num_mixes)
      out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[
        self.num_mixes * self.output_dim,
        self.num_mixes,
        self.num_mixes
      ], axis=-1, name='mdn_coef_split')

      def loss_i(i):
        batch_size = tf.shape(out_sigma)[0]
        sigma_i = tf.slice(out_sigma, [0, i], [batch_size, 1], name='mdn_sigma_slice')
        pi_i = tf.slice(out_pi, [0, i], [batch_size, 1], name='mdn_pi_slice')
        mu_i = tf.slice(out_mu, [0, i * self.output_dim], [batch_size, self.output_dim], name='mdn_mu_slice')
        dist = tf.distributions.Normal(loc=mu_i, scale=sigma_i)
        loss = dist.prob(y_true) # find the pdf around each value in y_true
        loss = pi_i * loss
        return loss

      result = tf.map_fn(lambda  m: loss_i(m), mix, dtype=tf.float32, name='mix_map_fn')
      result = tf.reduce_sum(result, axis=0, keep_dims=False)
      result = -tf.log(result)
      result = tf.reduce_mean(result)
      return result

    if self.kernel == 'unigaussian':
      with tf.name_scope('MDNLayer'):
        return unigaussian_loss