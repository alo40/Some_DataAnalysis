import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import math


def calculate_binomial_probability(n, k, p):
    return comb(n, k) * (p ** k) * (1 - p) ** (n - k)


def calculate_poisson_probability(L, k):
    return np.exp(-L) * (L ** k) / math.factorial(k)


# Parameters
n = 31000
k = 63
p = k / n
L = p * n  # lambda
N = k + 50  # total loop iterations

# Single Binomial case
f = comb(n, k) * (p ** k) * (1 - p) ** (n - k)

# Binomial case
pmf_binomial = np.zeros(N, dtype=float)
for i in range(0, N):
    pmf_binomial[i] = calculate_binomial_probability(n, i, p)

# Poisson case
pmf_poisson = np.zeros(N, dtype=float)
for i in range(0, N):
    pmf_poisson[i] = calculate_poisson_probability(L, i)

# Plotting
plt.plot(pmf_binomial, label="Binomial")
plt.plot(pmf_poisson, linestyle="None", markerfacecolor='None', marker='o', label="Poisson")
plt.legend()
plt.grid()
plt.title(f"Comparison Binomial vs Poisson distributions | n = {n} | k = {k} | p = {p:.4} | lambda = {L:.4} |")
# plt.title(f"Probability of {k} ocurrences in {n} cases is {f:.4}")  # not used
plt.xlabel("Number of patients")
plt.ylabel("Probability of ocurrence")
plt.show()