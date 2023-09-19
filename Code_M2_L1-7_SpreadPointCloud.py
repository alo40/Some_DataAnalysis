import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import axes3d
from myLibrary import calculate_projection_vector


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# projection vector
ux = 1 / np.sqrt(5) * 1
uy = 1 / np.sqrt(5) * 2

# vectors
x0 = [ 0,  0,  0]
y0 = [ 0,  0,  0]
z0 = [ 0,  0,  0]
x1 = [ 1,  3, -1]
y1 = [ 2,  4,  0]
z1 = [ 0,  0,  0]

n = 2  # select vector
for i in [n]:
    ax.plot([x0[i], x1[i]], [y0[i], y1[i]], zs=[z0[i], z1[i]], marker='o', label=f"vector {i}")

# projected vectors
x = np.array([x1[n], y1[n]])
u = np.array([ux, uy])
v = calculate_projection_vector(x, u)
print(f"projected vector = {v} of vector {n}")

ax.plot([0, u[0]], [0, u[1]], zs=[0, 0], marker='o', label="unit vector")
ax.plot([0, v[0]], [0, v[1]], zs=[0, 0], marker='o', label="projection")

ax.set_xlabel('X-Axis')
ax.set_ylabel('Y-Axis')
ax.set_zlabel('Z-Axis')

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(0, 5)

ax.view_init(elev=90, azim=-90)
plt.legend()
plt.show()
