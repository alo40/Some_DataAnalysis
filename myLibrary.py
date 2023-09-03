from scipy.special import comb
import math


def calculate_binomial_probability(n, k, p):
    return comb(n, k) * (p ** k) * (1 - p) ** (n - k)


def calculate_poisson_probability(L, k):
    return math.exp(-L) * (L ** k) / math.factorial(k)


def calculate_hypergeometric_distribution(n1, n2, k1, k2):
    return comb(n1, k1) * comb(n2, k2) / comb((n1 + n2), (k1 + k2))