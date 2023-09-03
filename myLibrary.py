from scipy.special import comb
import math


def calculate_binomial_probability(n, k, p):
    return comb(n, k) * (p ** k) * (1 - p) ** (n - k)


def calculate_poisson_probability(L, k):
    return math.exp(-L) * (L ** k) / math.factorial(k)
