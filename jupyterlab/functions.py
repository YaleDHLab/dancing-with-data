import numpy as np
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

def get_plot(df, axis_min=0, axis_max=1, frames=200, speed=45, start_time_index=0, run_tests=False):
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

def load_data(file):
    # X.shape = n_body_parts, n_time_intervals, n_dimensions (3)
    X = np.load(file)
    
    # store the shapes of X's values
    n_vertices, n_time, n_dims = X.shape

    # labels[i] is a label for the ith body_part
    labels = ['ARIEL.position', 'C7.position', 'CLAV.position', 'LANK.position', 'LBHD.position', 'LBSH.position', 'LBWT.position', 'LELB.position', 'LFHD.position', 'LFRM.position', 'LFSH.position', 'LFWT.position', 'LHEL.position', 'LIEL.position', 'LIHAND.position', 'LIWR.position', 'LKNE.position', 'LKNI.position', 'LMT1.position', 'LMT5.position', 'LOHAND.position', 'LOWR.position', 'LSHN.position', 'LTHI.position', 'LTOE.position', 'LUPA.position', 'LabelingHips.position', 'MBWT.position', 'MFWT.position', 'RANK.position', 'RBHD.position', 'RBSH.position', 'RBWT.position', 'RELB.position', 'RFHD.position', 'RFRM.position', 'RFSH.position', 'RFWT.position', 'RHEL.position', 'RIEL.position', 'RIHAND.position', 'RIWR.position', 'RKNE.position', 'RKNI.position', 'RMT1.position', 'RMT5.position', 'ROHAND.position', 'ROWR.position', 'RSHN.position', 'RTHI.position', 'RTOE.position', 'RUPA.position', 'STRN.position', 'SolvingHips.position', 'T10.position']
    
#     flat = flatten(X)
#     unflat = unflatten(flat)
    return(X)

from math import floor

# define functions to flatten and unflatten data

def flatten(df, run_tests=False):
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

def unflatten(df, run_tests=False, start_time_index=0):
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

