import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
b = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0,0,1]
])

c = a@b

print(c)