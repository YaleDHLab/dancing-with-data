#!/usr/bin/env python
# coding: utf-8


import os, subprocess
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sys
sys.path.append('/project/hep/demers/mnp3/AI/dancing-with-robots/')

# automatically refreshes the MDN script if it changes
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# set a seed to control all randomness
from tensorflow import set_random_seed
from numpy.random import seed
set_random_seed(1)
seed(1)


# Load some data:

X = np.load('../data/npy/mariel_knownbetter.npy')
n_joints, n_timeframes, n_dims = X.shape
labels = ['ARIEL.position', 'C7.position', 'CLAV.position', 'LANK.position', 'LBHD.position', 'LBSH.position', 'LBWT.position', 'LELB.position', 'LFHD.position', 'LFRM.position', 'LFSH.position', 'LFWT.position', 'LHEL.position', 'LIEL.position', 'LIHAND.position', 'LIWR.position', 'LKNE.position', 'LKNI.position', 'LMT1.position', 'LMT5.position', 'LOHAND.position', 'LOWR.position', 'LSHN.position', 'LTHI.position', 'LTOE.position', 'LUPA.position', 'LabelingHips.position', 'MBWT.position', 'MFWT.position', 'RANK.position', 'RBHD.position', 'RBSH.position', 'RBWT.position', 'RELB.position', 'RFHD.position', 'RFRM.position', 'RFSH.position', 'RFWT.position', 'RHEL.position', 'RIEL.position', 'RIHAND.position', 'RIWR.position', 'RKNE.position', 'RKNI.position', 'RMT1.position', 'RMT5.position', 'ROHAND.position', 'ROWR.position', 'RSHN.position', 'RTHI.position', 'RTOE.position', 'RUPA.position', 'STRN.position', 'SolvingHips.position', 'T10.position']
print(X.shape) # (number of joints) X (number of time frames) X (x,y,z dimensions)


from math import floor

# define functions to flatten and unflatten data

def flatten(df, run_tests=True):
  '''
  df is a numpy array with the following three axes:
    df.shape[0] = the index of a vertex
    df.shape[1] = the index of a time stamp
    df.shape[2] = the index of a dimension (x, y, z)
  
  So df[1][0][2] is the value for the 1st vertex (0-based) at time 0 in dimension 2 (z).
  
  To flatten this dataframe will mean to push the data into shape:
    flattened.shape[0] = time index
    flattened.shape[1] = [vertex_index*3] + dimension_vertex
    
  So flattened[1][3] will be the 3rd dimension of the 1st index (0-based) at time 1. 
  '''
  if run_tests:
    assert df.shape == X.shape and np.all(df == X)
  
  # reshape X such that flattened.shape = time, [x0, y0, z0, x1, y1, z1, ... xn-1, yn-1, zn-1]
  flattened = X.swapaxes(0, 1).reshape( (df.shape[1], df.shape[0] * df.shape[2]), order='C' )

  if run_tests: # switch to false to skip tests
    for idx, i in enumerate(df):
      for jdx, j in enumerate(df[idx]):
        for kdx, k in enumerate(df[idx][jdx]):
          assert flattened[jdx][ (idx*df.shape[2]) + kdx ] == df[idx][jdx][kdx]
          
  return flattened

def unflatten(df, run_tests=True, start_time_index=0):
  '''
  df is a numpy array with the following two axes:
    df.shape[0] = time index
    df.shape[1] = [vertex_index*3] + dimension_vertex
    
  To unflatten this dataframe will mean to push the data into shape:
    unflattened.shape[0] = the index of a vertex
    unflattened.shape[1] = the index of a time stamp
    unflattened.shape[2] = the index of a dimension (x, y, z)
    
  So df[2][4] == unflattened[1][2][0]
  '''
  if run_tests:
    assert (len(df.shape) == 2) and (df.shape[1] == X.shape[0] * X.shape[2])
  
  unflattened = np.zeros(( X.shape[0], df.shape[0], X.shape[2] ))

  for idx, i in enumerate(df):
    for jdx, j in enumerate(df[idx]):
      kdx = int(floor(jdx / 3))
      ldx = int(jdx % 3)
      unflattened[kdx][idx][ldx] = df[idx][jdx]

  if run_tests: # set to false to skip tests
    for idx, i in enumerate(unflattened):
      for jdx, j in enumerate(unflattened[idx]):
        for kdx, k in enumerate(unflattened[idx][jdx]):
          assert( unflattened[idx][jdx][kdx] == X[idx][int(start_time_index)+jdx][kdx] )

  return unflattened

flat = flatten(X)
unflat = unflatten(flat)


import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.art3d import juggle_axes
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib import animation
from copy import deepcopy
import matplotlib

# ask matplotlib to plot up to 2^128 frames in animations
matplotlib.rcParams['animation.embed_limit'] = 2**128

def update_points(time, points, df):
  '''
  Callback function called by plotting function below. Mutates the vertex
  positions of each value in `points` so the animation moves
  @param int time: the index of the time slice to visualize within `df`
  @param mpl_toolkits.mplot3d.art3d.Path3DCollection points: the actual
    geometry collection whose internal values this function mutates to move
    the displayed points
  @param numpy.ndarray df: a numpy array with the following three axes:
    df.shape[0] = n_vertices
    df.shape[1] = n_time_slices
    df.shape[2] = n_dimensions
  '''
  points._offsets3d = juggle_axes(df[:,time,0], df[:,time,1], df[:,time,2], 'z')

def get_plot(df, axis_min=0, axis_max=1, frames=200, speed=45, start_time_index=0, run_tests=True):
  '''
  General function that can plot numpy arrays in either of two shapes.
  @param numpy.ndarray df: a numpy array with either of the following two shapes:
    Possibility one:
      df.shape[0] = n_vertices
      df.shape[1] = n_time_slices
      df.shape[2] = n_dimensions
    Possibility two:
      df.shape[0] = n_time_slices
      df.shape[1] = [x0, y0, z0, x1, y1, z1, ... xn-1, yn-1, zn-1]
    If the latter is received, we "unflatten" the df into the three dimensional variant
  @param int axis_min: the minimum value of each axis scale
  @param int axis_max: the maximum value of each axis scale
  @param int frames: the number of time slices to animate.
  @param int speed: the temporal duration of each frame. Increase to boost fps.
  @param int start_time_index: the index position of the first frame in df within X. In other
    words, if df starts at the nth time frame from X, start_time_index = n.
  @param bool run_tests: boolean indicating whether we'll run the data validation
    tests, should we need to unflatten the array. Should be set to False if we're passing
    in predicted values, as they'll differ from X values.
  '''
  df = deepcopy(df)
  if len(df.shape) == 2:
    df = unflatten(df, start_time_index=start_time_index, run_tests=run_tests)
  # center the data for visualization
  df -= np.amin(df, axis=(0, 1))
  df /= np.amax(df, axis=(0, 1))
  # scale the z dimension
  df[:,:,2] *= -1
  df[:,:,2] += 1  
  # plot the data
  fig = plt.figure()
  ax = p3.Axes3D(fig)
  ax.set_xlim(axis_min, axis_max)
  ax.set_ylim(axis_min, axis_max)
  ax.set_zlim(axis_min, axis_max*1.5)
  points = ax.scatter(df[:,0,0], df[:,0,1], df[:,0,2], depthshade=False) # x,y,z vals
  return animation.FuncAnimation(fig,
    update_points,
    frames,
    interval=speed,
    fargs=(points, df),
    blit=False  
  ).to_jshtml()

# HTML(get_plot(unflat, frames=10, start_time_index=600))


#  Reduce dimensionality

### Move each frame to (x,y)=(0,0), leaving the z dimension free
print(X.shape)
X[:,:,:2] -= X[:,:,:2].mean(axis=0, keepdims=True)

### Then "flatten" dimensions, i.e. instead of n timestamps x 55 joints x 3 dimensions, use n timestamps x 165 joints
print(X.shape)
# reshape such that flattened.shape = time, [x0, y0, z0, x1, y1, z1, ... xn-1, yn-1, zn-1]
flat = X.swapaxes(0, 1).reshape( (X.shape[1], X.shape[0] * X.shape[2]), order='C' )
print(flat.shape)

column_names = [ 'joint'+str(i)+'_'+str(j) for i in range(int(flat.shape[1]/3)) for j in ['x','y','z']]
print(column_names)

df = pd.DataFrame(flat, columns=column_names)
print('Size of the dataframe: {}'.format(df.shape))
print(df)


# # ### Use PCA

# # In[93]:


# from sklearn.decomposition import PCA

# pca = PCA(n_components=2) # can either do this by num of desired components...
# # pca = PCA(.95) # ...or by percentage variance you want explained 

# pca_columns=[]

# pca_result = pca.fit_transform(df.values)
# for i in range(pca_result.shape[1]):
#     df['pca_'+str(i)] = pca_result[:,i]
#     pca_columns.append('pca_'+str(i))
    
# print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# # Plot just the first two dimensions in PCA space
# plt.scatter(df.loc[:,"pca_0"],df.loc[:,"pca_1"],c=np.arange(len(df)),cmap='viridis')
# plt.xlabel("pca_0")
# plt.ylabel("pca_1")


# # In[94]:


# # Get the transformed dataset 
# print(df[pca_columns].shape)
# df[pca_columns]


# # # Split into training datasets

# # In[95]:


# pca_data = df[pca_columns]
# pca_data.shape


# # In[96]:


# pca_data.loc[0:1,:]


# # In[97]:


# # train_x has shape: n_samples, look_back, n_vertices*3
# look_back = 10 # number of previous time slices to use to predict the time positions at time `i`
# train_x = []
# train_y = []

# # each i is a time slice; these time slices start at idx `look_back` (so we can look back `look_back` slices)
# for i in range(look_back, n_timeframes-1, 1):
#     train_x.append( pca_data.loc[i-look_back:i-1].to_numpy() )
#     train_y.append( pca_data.loc[i] )
    
# train_x = np.array(train_x)
# train_y = np.asarray(train_y)

# print(train_x.shape)
# print(train_y.shape)


# # # Build the Model

# # In[98]:


# from utils.mdn import MDN
# from keras.models import Sequential, Model
# from keras.layers import Dense, LSTM, Dropout, Activation, CuDNNLSTM
# from keras.layers.advanced_activations import LeakyReLU
# from keras.losses import mean_squared_error
# from keras.optimizers import Adam
# from keras import backend as K
# import keras, os

# # config
# cells = [10, 10, 10, 10] # number of cells in each lstm layer
# output_dims = int(pca_data.shape[1]) # number of coordinate values to be predicted by each gaussian model
# input_shape = (look_back, output_dims) # shape of each input feature
# use_mdn = True # whether to use the MDN final layer or not
# n_mixes = 2 # number of gaussian models to build if use_mdn == True

# # optimizer params
# lr = 0.00001 # the learning rate of the model
# optimizer = Adam(lr=lr, clipvalue=0.5)

# # use tensorflow backend
# os.environ['KERAS_BACKEND'] = 'tensorflow'

# # determine the LSTM cells to use (hinges on whether GPU is available to keras)
# gpus = K.tensorflow_backend._get_available_gpus()
# LSTM_UNIT = CuDNNLSTM if len(gpus) > 0 else LSTM
# print('GPUs found:', gpus)

# # build the model
# model = Sequential()
# model.add(LSTM_UNIT(cells[0], return_sequences=True, input_shape=input_shape, ))
# model.add(LSTM_UNIT(cells[1], return_sequences=True, ))
# model.add(LSTM_UNIT(cells[2], ))
# model.add(Dense(cells[3]), )

# if use_mdn:
#     mdn = MDN(output_dims, n_mixes)
#     model.add(mdn)
#     model.compile(loss=mdn.get_loss_func(), optimizer=optimizer, metrics=['accuracy'])
# else:
#     model.add(Dense(output_dims, activation='tanh'))
#     model.compile(loss=mean_squared_error, optimizer=optimizer, metrics=['accuracy'])

# model.summary()


# # In[99]:


# # check untrained (baseline) accuracy
# model.evaluate(train_x, train_y)


# # # Train the model

# # In[117]:


# # from utils.logger import Logger


# # In[101]:


# from keras.callbacks import TerminateOnNaN
# from livelossplot import PlotLossesKeras
# from datetime import datetime
# import time, keras, os, json
  
# class Logger(keras.callbacks.Callback):
#   '''Save the model and its weights every `self.save_frequency` epochs'''
#   def __init__(self):
#     self.epoch = 0 # stores number of completed epochs
#     self.save_frequency = 1 # configures how often we'll save the model and weights
#     self.date = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M')
#     if not os.path.exists('snapshots'): os.makedirs('snapshots')
#     self.save_config()
    
#   def save_config(self):
#     with open('snapshots/' + self.date + '-config.json', 'w') as out:
#       json.dump({
#         'look_back': look_back,
#         'cells': cells,
#         'use_mdn': use_mdn,
#         'n_mixes': n_mixes,
#         'lr': lr,
#       }, out)
  
#   def on_batch_end(self, batch, logs={}, shape=train_x.shape):
#     if (batch+1 == shape[0]): # batch value is batch index, which is 0-based
#       self.epoch += 1
#       if (self.epoch > 0) and (self.epoch % self.save_frequency == 0):
#         path = 'snapshots/' + self.date + '-' + str(batch)
#         model.save(path + '.model')
#         model.save_weights(path + '.weights')

# #K.set_value(optimizer.lr, 0.00001)
# callbacks = [Logger(), TerminateOnNaN()]
# history = model.fit(train_x, train_y, epochs=1, batch_size=1, shuffle=False, callbacks=callbacks)


# # In[116]:


# from datetime import datetime
# model_path = '/project/hep/demers/mnp3/AI/dancing-with-robots/snapshots/'+datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

# # Save trained model
# model.save(model_path + '.model')
# model.save_weights(model_path + '.weights')

# # Load trained model
# model.load_weights(model_path + '.weights')


# # # Visualize Results
# # 
# # First let's analyze how well the model learned the input sequence
# # 
# # ### Assess Model Performance on Inputs

# # In[118]:


# # visualize how well the model learned the input sequence
# n_frames = 500 # n frames of time slices to generate
# frames = []

# test_x = train_x[:n_frames] # data to pass into forward prop through the model
# y_pred = model.predict(test_x) # output with shape (n_frames, (output_dims+2) * n_mixes )

# # partition out the mus, sigs, and mixture weights
# for i in range(n_frames):
#     y = y_pred[i].squeeze()
#     mus = y[:n_mixes*output_dims]
#     sigs = y[n_mixes*output_dims:n_mixes*output_dims + n_mixes]
#     alphas = y[-n_mixes:]

#     # find the most likely distribution - then disregard that number and use the first Gaussian :)
#     alpha_idx = np.argmax(alphas)
#     alpha_idx = 0

#     # pull out the mus that correspond to the selected alpha index
#     positions = mus[alpha_idx * output_dims:(alpha_idx+1) * output_dims]
#     frames.append(positions)

# frames = np.array(frames)


# # In[122]:


# # Convert from latent space to real space
# frames_real = np.dot(frames, pca.components_)
# print("Reconstructed shape", frames_real.shape)


# # In[123]:


# # skip the first look_back frames
# HTML(get_plot(frames_real, frames=n_frames, run_tests=False))


# # ### Assess Model's Ability to Generate New Sequences

# # In[124]:


# def softmax(x):
#   """Compute softmax values for each sets of scores in x."""
#   r = np.exp(x - np.max(x))
#   return r / r.sum()

# n_frames = 500 # n frames of time slices to generate
# frames = []

# seed = np.random.randint(0, len(train_x)-1)
# x = np.expand_dims(train_x[seed], axis=0)
# print(' * seeding with', seed)

# for i in range(n_frames):
#   y = model.predict(x).squeeze()
#   mus = y[:n_mixes*output_dims]
#   sigs = y[n_mixes*output_dims:-n_mixes]
#   alphas = softmax(y[-n_mixes:])
  
#   # select the alpha channel to use
#   alpha_idx = np.argmax(alphas)
  
#   # grab the mus and sigs associated with the selected alpha_idx
#   frame_mus = mus.ravel()[alpha_idx*output_dims : (alpha_idx+1)*output_dims]
#   frame_sig = sigs[alpha_idx] / 100
  
#   # now sample from each Gaussian
#   positions = [np.random.normal(loc=m, scale=frame_sig) for m in frame_mus]
#   positions = frame_mus
  
#   # add these positions to the results
#   frames.append(positions)
  
#   # pull out a new training example - stack the new result on
#   # all values after the first from the bottom-most value in the x's
#   start = x[:,1:,:]
#   end = np.expand_dims( np.expand_dims(positions, axis=0), axis=0 )
#   x = np.concatenate((start, end), axis=1)
  
# frames = np.array(frames)


# # In[125]:


# HTML(get_plot(np.dot(frames, pca.components_), frames=n_frames, run_tests=False))

