import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
# from mpl_toolkits.mplot3d import axes3d
from myLibrary import calculate_sample_cov_matrix


x1 = [0, 1]
x2 = [0, -1]
x = np.array([x1, x2])
n = x.shape[1]
# S = calculate_sample_cov_matrix(x)
S = 1 / n * x.T @ x
print(f"sample covariance matrix S = \n{S}")

eigenvalues, eigenvectors = LA.eig(S)
print(f"eigenvalues = \n{eigenvalues}")
print(f"eigenvectors = \n{eigenvectors}")

fig = plt.figure()
ax = fig.add_subplot()

ax.plot([x1[0], x1[1]], marker='o', label=f"vector x1")
ax.plot([x2[0], x2[1]], marker='o', label=f"vector x2")

ax.set_xlabel('X-Axis')
ax.set_ylabel('Y-Axis')

L = 2
ax.set_xlim(-L, L)
ax.set_ylim(-L, L)

plt.grid()
plt.legend()
plt.show()
