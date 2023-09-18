import numpy as np
from myLibrary import calculate_sample_cov_matrix

x1 = np.array([8, 4, 7])
x2 = np.array([2, 8, 1])
x3 = np.array([3, 1, 1])
x4 = np.array([9, 7, 4])
x = np.array([x1, x2, x3, x4])
S = calculate_sample_cov_matrix(x)

print(f"x = \n{x}")
print(f"The sample covariance matrix is: \n{S}")
