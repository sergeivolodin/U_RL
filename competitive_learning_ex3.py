import numpy as np

# numpy arr -> string
def np_str(arr):
  if len(arr.shape) == 1:
    return ', '.join([str(x) for x in arr])
  else:
    # column first order
    return ';\t'.join(['col%d: [%s]' % (i, np_str(x)) for i, x in enumerate(arr.T)])

# inputs (column vectors)
x = np.array([[0,1.,0], [1, 0, 1], [0, 1, 1]]).T

# inputs order (column ids)
order = [0,1,2,0,1,2]
#order = [0,1,0,1]

# initial weight vector (column vectors = neurons)
w = np.array([[1.5,0,0.5], [0, 0.5, 1.5]]).T

# learning rate
eta = 0.5

# going through all inputs
for i in order:
  # current input
  inp = x[:, i]

  # neuron activations
  activation = w.T @ inp

  # winner
  winner = np.argmax(activation)

  # winner update rule
  w[:, winner] += eta * (inp - w[:, winner])

  # printing result
  print('Input %d: %s,\tactivation %s,\twinner %d,\tweights %s' % (i, np_str(inp), np_str(activation), winner, np_str(w)))
