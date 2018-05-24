import numpy as np

n_states = 4
n_act = 2
eta = 0.1

# 0 back, 1 forw

beta = 0.1

Q = np.zeros((n_states, n_act))

def update_Q(s, a, r, s1, a1):
    Q[s, a] += eta * (r - (Q[s, a] - Q[s1, a1]))

def get_action(s):
    if s == 0: return 1
    if np.random.rand() <= beta:
        return np.random.choice(range(n_act))
    if Q[s, 0] == Q[s, 1]:
        return np.random.choice(range(n_act))
    res = np.argmax(Q[s, :])
    return res

def environment_step(s, a):
    r = 0
    s1 = s + (1 if a == 1 else -1)
    if s == 0 and a == 0:
        print('Error')
    if s == n_states - 1 and a == 1:
        print('Error')
    if s == n_states - 2 and a == 1:
        print('Reset')
        s1 = 0
        r = 1
    return s1, r

def arr_to_str(arr):
    return ', '.join(['%.2f' % x for x in arr])

s = 0
for i in range(10000):
    a = get_action(s)
    s1, r = environment_step(s, a)
    a1 = get_action(s1)
    print('SARSA: %d %d %d %d %d Q = %s' % (s, a, r, s1, a1, arr_to_str(Q.flatten())))
    update_Q(s, a, r, s1, a1)
    s = s1
