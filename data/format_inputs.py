import json
import math
import numpy as np
import os

input_files = ['carrie-10-mins.json']

for input_file in input_files:

  carrie = False
  mariel = False

  if 'carrie' in input_file.lower():
    carrie = True
  elif 'mariel' in input_file.lower():
    mariel = True
  else:
    print(' ! warning - check input orientation')

  j = json.load(open('json/' + input_file))

  # sort the keys (body part labels) to ensure consistent vertex order
  labels = sorted(j.keys())
  arr = []

  for i in labels:
    sub = []
    vals = j[i]
    _max = max([int(k) for k in vals])
    for k in range(_max+1):
      sub.append(vals[str(k)])
    arr.append(sub)

  # transpose the array because the json makes each body
  # part a row, but we want each body part to be a col
  X = np.array(arr).T

  # right now every three rows display the x, y, z vals for an observation
  # reshape so we have X[observation_idx][time_idx][dimension_idx]

  # is 0, 1, 2 for Mariel data, 0, 2, 1 for Carrie
  if carrie:
    print('axis order 0, 2, 1')
    axis_order = [0, 2, 1]
  elif mariel:
    print('axis order 0, 1, 2')
    axis_order = [0, 1, 2]

  old = X
  rows, cols = [int(i) for i in old.shape]
  X = np.zeros(( cols, int(rows/3), 3 ))
  for r_idx, row in enumerate(range(rows)):
    time = math.floor(r_idx / 3)
    coord = r_idx % 3
    for c_idx, col in enumerate(range(cols)):
      X[c_idx][time][axis_order[coord]] = old[r_idx][c_idx]

  if carrie:
    print('inverting z axis')
    X[:,:,2] *= -1
    X[:,:,2] += 1

  output_file = '.'.join(input_file.split('.')[:-1]) + '.npy'
  np.save('npy/' + output_file, X)

  print('saving', output_file)
  print('X shape', X.shape)
  print('labels:', list(labels))