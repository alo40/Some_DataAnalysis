import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from numpy import genfromtxt
from myLibrary import calculate_multilinear_OLS, add_intercept, cal_cost, gradient_descent


# def gradient_descent(gradient, start, learn_rate, n_iter=100, tolerance=1e-06):
#     vector = start
#     diff_array = np.zeros(n_iter)
#     for i in range(n_iter):
#         diff = -learn_rate * gradient(vector)
#         diff_array[i] = diff
#         if np.all(np.abs(diff) <= tolerance):
#             break
#         vector += diff
#     return vector, diff_array
#
#
# x_min, diff_array = gradient_descent(gradient=lambda v: 2 * v, start=-10.0, learn_rate=0.2)
#
# # x = np.arange(-10, 10, 0.1)
# # plt.plot(x, x ** 2, 'b-')
# # plt.plot(x_min, x_min ** 2, 'rx')
#
# plt.plot(diff_array, 'x')
# plt.show()


# def calculate_gradient_descent(X, y, step_size=0.0001, precision=1e-04, n_iter=100_000):
#     beta = calculate_multilinear_OLS(X, y)
#     loss_function = X @ beta.T
#     diff_array = np.zeros(n_iter)
#     for i in range(n_iter):
#         diff = -step_size * np.gradient(loss_function)
#         diff_array[i] = diff[1]
#         if np.all(np.abs(diff) <= precision):
#             print(i)
#             break
#         loss_function += diff
#     return loss_function, diff_array


# stopping condition
# if np.sqrt(np.sum(np.square((beta - beta_last)/beta))) < precision

X = genfromtxt('syn_X.csv', delimiter=',')
X = add_intercept(X)
y = genfromtxt('syn_y.csv', delimiter=',')

theta = np.random.rand(3, 1)

# alpha_1 = 0.9
for alpha in np.arange(0.001, 0.005, 0.001):
    theta_loop, cost_history, theta_history, convergence = gradient_descent(X, y, theta, learning_rate=alpha)
    print(f"learning rate = {alpha}, convergence after {convergence}\n")
# plt.plot(cost_history, '-r', label=f"learning rate = {alpha_1}")

# alpha_2 = 0.8
# theta_2, cost_history, theta_history = gradient_descent(X, y, theta, learning_rate=alpha_2)
# plt.plot(cost_history, '-g', label=f"learning rate = {alpha_2}")
#
# alpha_3 = 0.7
# theta_3, cost_history, theta_history = gradient_descent(X, y, theta, learning_rate=alpha_3)
# plt.plot(cost_history, '-b', label=f"learning rate = {alpha_3}")
#
# # plt.ylim(0, 1)
# plt.legend()
# plt.show()
