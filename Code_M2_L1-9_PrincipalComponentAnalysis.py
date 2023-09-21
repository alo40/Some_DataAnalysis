import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
# from mpl_toolkits.mplot3d import axes3d
from myLibrary import (calculate_sample_cov_matrix,
                       calculate_projection_vector,
                       calculate_projection_scalar,
                       calculate_sample_variance)

# # exercise 9
# x1 = [0, 1]
# x2 = [0, -1]

# # exercise 10
# x1 = np.array([1, 1/2])
# x2 = np.array([-1, -1/2])

# exercise 11
x1 = np.array([0, 2])
x2 = np.array([1, -1])
x3 = np.array([-1, -1])

x = np.array([x1, x2, x3])
S = calculate_sample_cov_matrix(x)
# n = x.shape[0]
# S = 1 / n * x.T @ x
print(f"sample covariance matrix S = \n{S}")

eigenvalues, eigenvectors = LA.eig(S)
print(f"eigenvalues = \n{eigenvalues}")
print(f"eigenvectors = \n{eigenvectors}")

v1 = eigenvectors[:, 1]
v2 = eigenvectors[:, 0]

p1 = calculate_projection_vector(x1, v1)
p2 = calculate_projection_vector(x2, v1)
p3 = calculate_projection_vector(x3, v1)

y1 = calculate_projection_scalar(x1, v1)
y2 = calculate_projection_scalar(x2, v1)
y3 = calculate_projection_scalar(x3, v1)
y = np.array([y1, y2, y3])
sigma_y = calculate_sample_variance(y)

print(f"signed length projection of x1 onto v1 is {y1}")
print(f"signed length projection of x2 onto v1 is {y2}")
print(f"signed length projection of x3 onto v1 is {y2}")
print(f"sample variance of projected data is {sigma_y}")

fig = plt.figure()
ax = fig.add_subplot()

ax.plot([0, x1[0]], [0, x1[1]], marker='o', label=f"vector x1")
ax.plot([0, x2[0]], [0, x2[1]], marker='o', label=f"vector x2")
ax.plot([0, x3[0]], [0, x3[1]], marker='o', label=f"vector x3")

ax.plot([0, v1[0]], [0, v1[1]], marker='v', label=f"eigenvector v1")
ax.plot([0, v2[0]], [0, v2[1]], marker='v', label=f"eigenvector v2")

m1 = y1*v1
ax.plot(m1[0], m1[1], marker='x', markersize=10, label=f"projection of x1 in v1")
m2 = y2*v1
ax.plot(m2[0], m2[1], marker='x', markersize=10, label=f"projection of x2 in v1")
m3 = y3*v1
ax.plot(m3[0], m3[1], marker='x', markersize=10, label=f"projection of x3 in v1")


ax.set_xlabel('X-Axis')
ax.set_ylabel('Y-Axis')

L = 2
ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.set_box_aspect(1)

plt.grid()
plt.legend()
plt.show()
