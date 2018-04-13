import numpy as np

M = 10
beta = 1.
d = 2

def B(i, j):
    if abs(i - j) <= d:
        return +1
    else: return -beta

def g(x):
    if x >= 0:
        return +1
    else:
        return 0

def y_new_i(y_old, i):
    s = 0.
    for j in range(M):
        s += B(i, j) * y_old[j]
    return g(s)

def y_new(y_old):
    y = np.zeros_like(y_old)
    for i in range(M):
        y[i] = y_new_i(y_old, i)
    return y

y = np.zeros(M)
y[2] = 1
for t in range(10):
    print(y)
    y = y_new(y)
