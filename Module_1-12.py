import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from myLibrary import calculate_binomial_probability, calculate_poisson_probability


# Parameters
n = 31000
k = 63
p = k / n
L = p * n  # lambda
N = k + 50  # total loop iterations

# Single Binomial case
f = calculate_binomial_probability(n, k, p)

# Binomial case
pmf_binomial = np.zeros(N, dtype=float)
for i in range(0, N):
    pmf_binomial[i] = calculate_binomial_probability(n, i, p)

# Poisson case
pmf_poisson = np.zeros(N, dtype=float)
for i in range(0, N):
    pmf_poisson[i] = calculate_poisson_probability(L, i)

# Numerical Integration
significance_index = 63
dx = 1  # spacing of integration points, default is 1
area_under_curve_simpson = simpson(pmf_poisson[:significance_index], dx=dx)
print(f"The area under the curve using simpson is {area_under_curve_simpson}")
#
# area_under_curve_trapz = np.trapz(pmf_poisson[:significance_index], dx=dx)  # not used
# print(f"The area under the curve using trapz is {area_under_curve_trapz}")

# Plotting
plt.plot(pmf_binomial, label="Binomial")
plt.plot(pmf_poisson, linestyle="None", markerfacecolor='None', marker='o', label="Poisson")
plt.plot([significance_index, significance_index], [0, 0.05], color='red', linestyle='dashed', label="significance line")
plt.xlim([0, N])
plt.legend()
plt.grid()
plt.title(f"Comparison Binomial vs Poisson distributions | n = {n} | k = {k} | p = {p:.4} | lambda = {L:.4} | area left to sig. line = {area_under_curve_simpson:.4}")
# plt.title(f"Probability of {k} ocurrences in {n} cases is {f:.4}")  # not used
plt.xlabel("Number of patients")
plt.ylabel("Probability of ocurrence")
plt.show()