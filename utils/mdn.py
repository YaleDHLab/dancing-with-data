# via https://github.com/cpmpercussion/keras-mdn-layer

'''
A Mixture Density Layer for Keras
cpmpercussion: Charles Martin (University of Oslo) 2018
https://github.com/cpmpercussion/keras-mdn-layer

Hat tip to [Omimo's Keras MDN layer](https://github.com/omimo/Keras-MDN) for a starting point for this code.
'''
import keras
from keras import backend as K
from keras.layers import Dense
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def elu_activation(x, plus_one=False):
  '''Exponential Linear Unit activation with a very small addition to help prevent NaN in loss.'''
  if plus_one:
    return (K.elu(x) + 1 + 1e-8)
  return (K.elu(x) + 1e-8)


class MDN(Layer):
  '''A Mixture Density Network Layer for Keras.
  This layer has a few tricks to avoid NaNs in the loss function when training:
    - Activation for variances is ELU + 1 + 1e-8 (to avoid very small values)
    - Mixture weights (pi) are trained in as logits, not in the softmax space.

  A loss function needs to be constructed with the same output dimension and number of mixtures.
  A sampling function is also provided to sample from distribution parametrised by the MDN outputs.
  '''

  def __init__(self, output_dimension, num_mixtures, **kwargs):
    self.output_dim = output_dimension
    self.num_mix = num_mixtures
    with tf.name_scope('MDN'):
      self.mdn_mus = Dense(self.num_mix * self.output_dim, name='mdn_mus') # mix*output vals, no activation
      self.mdn_sigmas = Dense(self.num_mix * self.output_dim, activation=elu_activation, name='mdn_sigmas') # mix*output vals exp activation
      self.mdn_pi = Dense(self.num_mix, name='mdn_pi') # mix vals, logits
    super(MDN, self).__init__(**kwargs)

  def build(self, input_shape):
    self.mdn_mus.build(input_shape)
    self.mdn_sigmas.build(input_shape)
    self.mdn_pi.build(input_shape)
    self.trainable_weights = self.mdn_mus.trainable_weights + self.mdn_sigmas.trainable_weights + self.mdn_pi.trainable_weights
    self.non_trainable_weights = self.mdn_mus.non_trainable_weights + self.mdn_sigmas.non_trainable_weights + self.mdn_pi.non_trainable_weights
    super(MDN, self).build(input_shape)

  def call(self, x, mask=None):
    with tf.name_scope('MDN'):
      mdn_out = keras.layers.concatenate([
        self.mdn_mus(x),
        self.mdn_sigmas(x),
        self.mdn_pi(x)
      ], name='mdn_outputs')
    return mdn_out

  def compute_output_shape(self, input_shape):
    '''Returns output shape, showing the number of mixture parameters.'''
    return (input_shape[0], (2 * self.output_dim * self.num_mix) + self.num_mix)

  def get_config(self):
    config = {
      'output_dimension': self.output_dim,
      'num_mixtures': self.num_mix
    }
    base_config = super(MDN, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def get_mixture_loss_func(output_dim, num_mixes):
  '''Construct a loss functions for the MDN layer parametrised by number of mixtures.'''
  # Construct a loss function with the right number of mixtures and outputs
  def loss_func(y_true, y_pred):
    # Reshape inputs in case this is used in a TimeDistribued layer
    y_pred = tf.reshape(y_pred, [-1, (2 * num_mixes * output_dim) + num_mixes], name='reshape_ypreds')
    y_true = tf.reshape(y_true, [-1, output_dim], name='reshape_ytrue')
    # Split the inputs into paramaters
    out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[
      num_mixes * output_dim,
      num_mixes * output_dim,
      num_mixes
    ], axis=-1, name='mdn_coef_split')
    # produces flat list that contains `num_mixes` instances of `output_dim` [n, n, n, ...]
    component_splits = [output_dim] * num_mixes
    # produces `num_mixes` arrays with the mus
    mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
    # produces `num_mixes` arrays with the sigs
    sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
    cat = tfd.Categorical(logits=out_pi)
    # produces num_mixes arrays each with a multivariate normal distribution with a single mu and sigma
    coll = [tfd.MultivariateNormalDiag(loc=mu, scale_diag=sig) for mu, sig in zip(mus, sigs)]
    mixture = tfd.Mixture(cat=cat, components=coll)
    loss = mixture.log_prob(y_true)
    loss = tf.negative(loss)
    loss = tf.reduce_mean(loss)
    return loss

  # Actually return the loss_func
  with tf.name_scope('MDN'):
    return loss_func


def split_mixture_params(params, output_dim, num_mixes):
  '''Splits up an array of mixture parameters into mus, sigmas, and pis
  depending on the number of mixtures and output dimension.'''
  mus = params[:num_mixes*output_dim]
  sigs = params[num_mixes*output_dim:2*num_mixes*output_dim]
  pi_logits = params[-num_mixes:]
  return mus, sigs, pi_logits


def softmax(w, t=1.0):
  '''Softmax function for a list or numpy array of logits. Also adjusts temperature.'''
  e = np.array(w) / t  # adjust temperature
  e -= e.max()  # subtract max to protect from exploding exp values.
  e = np.exp(e)
  dist = e / np.sum(e)
  return dist


def sample_from_output(params, output_dim, num_mixes, temp=1.0):
  '''Sample from an MDN output with temperature adjustment.'''
  mus = params[:num_mixes*output_dim]
  sigs = params[num_mixes*output_dim:2*num_mixes*output_dim]
  pis = softmax(params[-num_mixes:], t=temp)
  m = sample_from_categorical(pis)
  # Alternative way to sample from categorical:
  # m = np.random.choice(range(len(pis)), p=pis)
  mus_vector = mus[m*output_dim:(m+1)*output_dim]
  sig_vector = sigs[m*output_dim:(m+1)*output_dim] * temp  # adjust for temperature
  cov_matrix = np.identity(output_dim) * sig_vector
  sample = np.random.multivariate_normal(mus_vector, cov_matrix, 1, tol=1e-6)
  return sample


def sample_from_categorical(dist):
  '''Samples from a categorical model PDF.'''
  r = np.random.rand(1)  # uniform random number in [0,1]
  accumulate = 0
  for i in range(0, dist.size):
    accumulate += dist[i]
    if accumulate >= r:
      return i
  tf.logging.info('Error sampling mixture model.')
  return -1
