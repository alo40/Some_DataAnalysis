import numpy as np
import numpy.linalg as LA

x1 = np.array([1, 1])
x2 = np.array([1, -1])
x3 = np.array([-1, 1])
x = np.array([x1, x2, x3])
B = x @ x.T
print(f"The Gram matrix for this dataset is \n{B}")

eigenvalues, eigenvectors = LA.eig(B)
print(f"eigenvalues \n{eigenvalues}")
print(f"eigenvectors \n{eigenvectors}")

v1 = eigenvectors[:, 0]
s1 = eigenvalues[0]
y = v1 * np.sqrt(s1)
print(f"The lower-dimensional embedding is \n{y}")