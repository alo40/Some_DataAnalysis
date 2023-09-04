import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

X = np.array([0.9, -0.9, 4.3, 2.9, 1.2, 3.0, 2.7, 0.6, 3.6, -0.5])
n = X.size
X_sample_mean = sum(X) / n
variance_unbiased = 1 / (n - 1) * sum((X - X_sample_mean) ** 2)
T = X_sample_mean / np.sqrt(variance_unbiased / n)

# using T value
p_value = 1 - t.cdf(T, n - 1)  # tail on the right side
print(f"The p-value of T={T} is equal to {p_value}")

# # check variance_unbiased value
# sigma_sum_sum = sum((X - X_sample_mean) ** 2)
# sigma_sum_iter = 0.
# for i in range(0, 10):
#     # print(f"{X[i]}")
#     sigma_sum_iter = sigma_sum_iter + (X[i] - X_sample_mean) ** 2
# print(f"Sigma sum using iteration: {sigma_sum_iter}")
# print(f"Sigma sum using sum method: {sigma_sum_sum}")
