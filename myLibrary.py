import numpy as np
from scipy.special import comb, factorial
from scipy.stats import expon


def find_nearest_value(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def calculate_binomial_probability(n, k, p):
    return comb(n, k) * (p ** k) * (1 - p) ** (n - k)


def calculate_poisson_probability(L, k):
    # return expon(-L) * (L ** k) / factorial(k)
    return np.exp(-L) * (L ** k) / factorial(k)


def calculate_hypergeometric_distribution(n1, n2, k1, k2):
    return comb(n1, k1) * comb(n2, k2) / comb((n1 + n2), (k1 + k2))


def calculate_sample_variance(x):
    return 1 / (x.size - 1) * ((x - x.mean()) ** 2).sum()


def calculate_sample_covariance(x, y):
    return ((x - x.mean()) * (y - y.mean())).sum() / (x.size - 1)


def calculate_sample_correlationCoeff(x, y):
    cov_xy = calculate_sample_covariance(x, y)
    var_x = calculate_sample_variance(x)
    var_y = calculate_sample_variance(y)
    return cov_xy / np.sqrt(var_x * var_y)


def calculate_predictor_coefficients(x, y):
    r = calculate_sample_correlationCoeff(x, y)
    s_x = np.sqrt(calculate_sample_variance(x))
    s_y = np.sqrt(calculate_sample_variance(y))
    beta_hat_1 = r * s_y / s_x
    beta_hat_0 = y.mean() - beta_hat_1 * x.mean()
    return beta_hat_1, beta_hat_0


def calculate_residuals(y, y_hat):
    return y - y_hat


def calculate_multilinear_OLS(x, y):
    beta_hat = np.linalg.inv(x.T @ x) @ x.T @ y
    return beta_hat


def add_intercept(X):
    return np.concatenate((np.ones_like(X[:,:1]), X), axis=1)


def calculate_multilinear_T_test(x, y):
    N = x.shape[0]  # data measurements
    p = x.shape[1]  # parameters
    beta_hat = calculate_multilinear_OLS(x, y)
    variance_hat = ((y - x @ beta_hat) ** 2).sum() / (N - p)
    sigma_squared = np.linalg.inv(x.T @ x).diagonal()
    return beta_hat / np.sqrt(variance_hat * sigma_squared)


def cal_cost(theta, X, y):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / 2 * m) * np.sum(np.square(predictions - y))
    return cost


def gradient_descent(X, y, theta, learning_rate=0.01, iterations=1_000_000):
    m = len(y)
    convergence = 0
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, X.shape[1]))
    for i in range(iterations):
        prediction = np.dot(X, theta)
        theta = theta - (1 / m) * learning_rate * X.T.dot(prediction - y)
        theta_history[i, :] = theta.T[0]
        cost_history[i] = cal_cost(theta, X, y)
        if cost_history[i] <= 1e2:
            print(f"Convergence at step {i}")
            convergence = i
            break
    return theta, cost_history, theta_history, convergence


def calculate_vector_sample_mean(x):
    n = x.shape[0]
    I = np.ones(n)
    return 1 / n * x.T @ I


def calculate_sample_cov_matrix(x):
    n = x.shape[0]
    H = calculate_orthogonal_matrix_H(x)
    return 1 / n * x.T @ H @ x
    # x_mean = calculate_vector_sample_mean(x)
    # outer_sum = 0
    # for i in range(n):
    #     outer_sum += np.outer(x[i], x[i].T)
    # return 1 / n * outer_sum - np.outer(x_mean, x_mean.T)


def calculate_orthogonal_matrix_H(x):
    n = x.shape[0]
    I = np.ones(n)
    In = np.identity(n)
    return In - 1 / n * np.outer(I, I.T)


def calculate_projection_vector(x, u):
    return np.outer(np.dot(u, x), u)[0]


def calculate_projection_scalar(x, u):
    p = calculate_projection_vector(x, u)
    return p @ u
    # return u / np.linalg.norm(u) @ x  # also valid


def calculate_ResidualSquareSum(x, y, a, b, c):
    d = (np.abs(a * x + b * y + c) / np.sqrt(a ** 2 + b ** 2)) ** 2
    return d.sum()

# NOT USED #############################################################
    # sigma_sum = 0
    # for i in range(x.size):
    #     sigma_sum += (x[i] - x.mean()) * (y[i] - y.mean())
    # return 1 / (x.size - 1) * sigma_sum

    # try:
    #     if not x.size == y.size:
    #         raise ValueError
    # except ValueError:
    #      print("Input vectors are not the same size")

    # sum = 0
    # for i in range(N):
    #     sum += (y[i] - x[i] @ beta_hat) ** 2
    # variance_hat = sum / (N - p)