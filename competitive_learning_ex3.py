import numpy as np
x = np.array([[0,1.,0], [1, 0, 1], [0, 1, 1]]).T
#order = [0,1,2,0,1,2]
order = [0,1,0,1]
w = np.array([[1.5,0,0.5], [0, 0.5, 1.5]]).T
eta = 0.5

for i in order:
  inp = x[:, i]
  activation = w.T @ inp
  winner = np.argmax(activation)
  w[:, winner] += eta * (inp - w[:, winner])
  print('Input %d %s, activation %s, winner %d, weights %s' % (i, str(inp), str(activation), winner, str(w)))
