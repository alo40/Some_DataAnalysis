from myLibrary import calculate_sample_covariance, calculate_sample_variance
import numpy as np
import matplotlib.pyplot as plt

# # example data set 1
# x = np.array([0, 2, 4, 5, 7, 8, 10])
# y = np.array([4, 3, 1, 1, 2, 1, 0])

# # example data set 2
# x = np.array([2, 4, 6, 8, 10])
# y = np.array([12, 11, 8, 3, 1])

x = np.array([0.0339, 0.0423, 0.213, 0.257, 0.273, 0.273, 0.450, 0.503, 0.503, \
0.637, 0.805, 0.904, 0.904, 0.910, 0.910, 1.02, 1.11, 1.11, 1.41, \
1.72, 2.03, 2.02, 2.02, 2.02])

y = np.array([-19.3, 30.4, 38.7, 5.52, -33.1, -77.3, 398.0, 406.0, 436.0, 320.0, 373.0, \
93.9, 210.0, 423.0, 594.0, 829.0, 718.0, 561.0, 608.0, 1.04E3, 1.10E3, \
840.0, 801.0, 519.0])

print(f"x mean = {x.mean()}")
print(f"y mean = {y.mean()}")

print(f"x sample standard deviation = {np.sqrt(calculate_sample_variance(x))}")
print(f"y sample standard deviation = {np.sqrt(calculate_sample_variance(y))}")

cov_xy = calculate_sample_covariance(x, y)
print(f"my covariance = {cov_xy}")
# print(f"numpy covariance = {np.cov(x, y)}")  # for testing

plt.plot(x, y, 'o')
plt.show()