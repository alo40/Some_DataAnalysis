import numpy as np
from scipy.special import comb, factorial
from scipy.stats import expon


def calculate_binomial_probability(n, k, p):
    return comb(n, k) * (p ** k) * (1 - p) ** (n - k)


def calculate_poisson_probability(L, k):
    return expon(-L) * (L ** k) / factorial(k)


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
