import json
import math
import numpy as np

j = json.load(open('json/dance.json'))

labels = j.keys()
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
old = X
rows, cols = [int(i) for i in old.shape]
X = np.zeros(( cols, int(rows/3), 3 ))
for r_idx, row in enumerate(range(rows)):
  time = math.floor(r_idx / 3)
  coord = r_idx % 3
  for c_idx, col in enumerate(range(cols)):
    X[c_idx][time][coord] = old[r_idx][c_idx]

# center each of the 3 dimensional features
X -= np.amin(X, axis=(0, 1))
X /= np.amax(X, axis=(0, 1))

# flip the sign on the dimension matplotlib considers the vertical
# dimension so the figure stands properly
X[:,:,2] *= -1

np.save('npy/dance.npy', X)

print('X shape', X.shape)
print('labels:', list(labels))