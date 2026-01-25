import time

def measure_time(func, args, repeat=1):
    total_elapse = 0

    for _ in range(repeat):
        st = time.time()
        if isinstance(func, list):
            for f, a in zip(func, args):
                f(*a)
        else:
            func(*args)
        elapse = time.time() - st
        total_elapse += elapse
    
    avg = total_elapse / repeat
    return avg

    
import matplotlib.pyplot as plt
import numpy as np


# plt.subplot([2, 1, 1])

a = np.array([1, 2, 3, 6])
b = np.array([5, 6, 7, 8])

plt.bar(a, b)

plt.show()
