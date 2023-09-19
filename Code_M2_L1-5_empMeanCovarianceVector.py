import numpy as np
from myLibrary import calculate_sample_cov_matrix, calculate_orthogonal_matrix_H


# 5
x1 = np.array([8, 4, 7])
x2 = np.array([2, 8, 1])
x3 = np.array([3, 1, 1])
x4 = np.array([9, 7, 4])
x = np.array([x1, x2, x3, x4])
S = calculate_sample_cov_matrix(x)
print(f"x = \n{x}")
print(f"The sample covariance matrix is: \n{S}")


# 6
x = np.array([2, -1, -2]).T
H = calculate_orthogonal_matrix_H(x)
Hx = H @ x
H2x = H @ H @ x
print(f"Hx = \n{Hx}")
print(f"H2x = \n{H2x}")
