import numpy as np
import matplotlib
import matplotlib.pyplot as plt


xs = [32, 64, 128, 256, 512]
times = [
    0.01674220561981201,
    0.006081938743591309,
    0.018069887161254884,
    0.03131325244903564,
    0.05104470252990723,
]

# times = [ 1, 2, 3, 4, 5]

x = np.array(xs)
y = np.array(times)

plt.plot(x, y)
plt.savefig("demo1")
