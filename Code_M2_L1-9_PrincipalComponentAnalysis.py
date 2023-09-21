import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
# from mpl_toolkits.mplot3d import axes3d
from myLibrary import calculate_sample_cov_matrix, calculate_projection_vector, calculate_projection_scalar

# # exercise 9
# x1 = [0, 1]
# x2 = [0, -1]

# exercise 10
x1 = np.array([1, 1/2])
x2 = np.array([-1, -1/2])

x = np.array([x1, x2])
n = x.shape[1]
S = calculate_sample_cov_matrix(x)
# S = 1 / n * x.T @ x
print(f"sample covariance matrix S = \n{S}")

eigenvalues, eigenvectors = LA.eig(S)
print(f"eigenvalues = \n{eigenvalues}")
print(f"eigenvectors = \n{eigenvectors}")

v1 = eigenvectors[:, 0]
v2 = eigenvectors[:, 1]

p1 = calculate_projection_vector(x1, v1)
p2 = calculate_projection_vector(x2, v1)

y1 = calculate_projection_scalar(x1, v1)
y2 = calculate_projection_scalar(x2, v1)

print(f"y1 = {y1}")
print(f"y2 = {y2}")

fig = plt.figure()
ax = fig.add_subplot()

ax.plot([0, x1[0]], [0, x1[1]], marker='o', label=f"vector x1")
ax.plot([0, x2[0]], [0, x2[1]], marker='o', label=f"vector x2")

ax.plot([0, v1[0]], [0, v1[1]], marker='o', label=f"eigenvector v1")
ax.plot([0, v2[0]], [0, v2[1]], marker='o', label=f"eigenvector v2")

m1 = y1*v1
ax.plot(m1[0], m1[1], marker='x', markersize=10, label=f"projection of x1 in v1")
m2 = y2*v1
ax.plot(m2[0], m2[1], marker='x', markersize=10, label=f"projection of x2 in v1")

ax.set_xlabel('X-Axis')
ax.set_ylabel('Y-Axis')

L = 2
ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.set_box_aspect(1)

plt.grid()
plt.legend()
plt.show()
