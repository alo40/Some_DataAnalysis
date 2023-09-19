import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import axes3d


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# vectors
x0 = [0, 0, 0, 0]
y0 = [0, 0, 0, 0]
z0 = [0, 0, 0, 0]
x1 = [1, 2, 1, 6]
y1 = [3, 1, 2, 7]
z1 = [1, 0, 4, 9]

for i in range(4):
    ax.plot([x0[i], x1[i]], [y0[i], y1[i]], zs=[z0[i], z1[i]])
plt.show()