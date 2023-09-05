from myLibrary import (calculate_sample_correlationCoeff,
                       calculate_sample_variance,
                       calculate_predictor_coefficients)
import numpy as np
import matplotlib.pyplot as plt


# exercise data set
x = np.array([0.0339, 0.0423, 0.213, 0.257, 0.273, 0.273, 0.450, 0.503, 0.503, \
0.637, 0.805, 0.904, 0.904, 0.910, 0.910, 1.02, 1.11, 1.11, 1.41, \
1.72, 2.03, 2.02, 2.02, 2.02])

y = np.array([-19.3, 30.4, 38.7, 5.52, -33.1, -77.3, 398.0, 406.0, 436.0, 320.0, 373.0, \
93.9, 210.0, 423.0, 594.0, 829.0, 718.0, 561.0, 608.0, 1.04E3, 1.10E3, \
840.0, 801.0, 519.0])

# optimized parameters for predictor
beta_hat_1, beta_hat_0 = calculate_predictor_coefficients(x, y)

print(f"beta_hat_1 = {beta_hat_1}")
print(f"beta_hat_0 = {beta_hat_0}")

plt.plot(x, y, 'o')
plt.plot(x, beta_hat_1 * x + beta_hat_0, 'red')
plt.show()