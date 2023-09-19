import numpy as np
from myLibrary import calculate_sample_cov_matrix

x1 = [1, 2]
x2 = [3, 4]
x3 = [-1, 0]
u = np.array([1 / np.sqrt(5), 2 / np.sqrt(5)])

x = np.array([np.dot(u, x1), np.dot(u, x2), np.dot(u, x3)])
S = calculate_sample_cov_matrix(x)
print(f"empirical variance S = {S}")

A = u.T * S @ u
print(f"other value = {A}")