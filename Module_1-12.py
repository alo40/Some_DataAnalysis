import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt


def calculate_binomial_probability(n, k, p):
    return comb(n, k) * (p ** k) * (1 - p) ** (n - k)


# Single case
n = 31000
k = 63
p = k / n
f = comb(n, k) * (p ** k) * (1 - p) ** (n - k)

# Loop case
N = k + 50
pmf = np.zeros(N, dtype=float)
for i in range(0, N):
    pmf[i] = calculate_binomial_probability(n, i, p)
    # print(pmf[i])

# Plotting
plt.plot(pmf)
plt.grid()
plt.title(f"Probability of {k} ocurrences in {n} cases is {f:.4}")
plt.xlabel("Number of patients")
plt.ylabel("Probability of ocurrence")
plt.show()