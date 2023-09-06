import numpy as np
import matplotlib.pyplot as plt
from myLibrary import calculate_sample_correlationCoeff, calculate_residuals, calculate_predictor_coefficients
import statsmodels.api as sm

# # example
# alpha = 1
# beta = 1
# x = np.arange(0, 4, 0.2)
# y = alpha * np.exp(beta * x)
# y_lin = beta * x + np.log(alpha)

# data from exercise
x = np.array([ 0.387, 0.723, 1.00, 1.52, 5.20, 9.54, 19.2, 30.1, 39.5 ])
y = np.array([ 0.241, 0.615, 1.00, 1.88, 11.9, 29.5, 84.0, 165.0, 248 ])

# transformation
x = np.log(x)
y = np.log(y)

r_xy = calculate_sample_correlationCoeff(x, y)
print(f"correlation coefficient xy = {r_xy}")

beta_1, beta_0 = calculate_predictor_coefficients(x, y)
y_hat = beta_1 * x + beta_0
e = calculate_residuals(y, y_hat)
print(f"beta_1 = {beta_1}")
print(f"beta_0 = {beta_0}")

plt.plot(x, y, '-*', label='y')
# plt.plot(x, e, 'x', label='residuals vs x')
# plt.plot(y_hat, e, 'o', label='residuals vs y_hat')
# plt.plot(x, y_lin, '-o', label='y_lin')

# sm.qqplot(x, line='s', label='x distribution')
# plt.legend()
# sm.qqplot(y, line='s', label='y distribution')
# plt.legend()

plt.legend()
plt.show()