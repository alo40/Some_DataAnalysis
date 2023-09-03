import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from myLibrary import calculate_poisson_probability


# Parameters for null hypothesis H0
n_H0 = 31000
k_H0 = 63
p_H0 = k_H0 / n_H0
L_H0 = p_H0 * n_H0  # lambda

# Parameters for alternative hypothesis HA
n_HA = 31000
k_HA = 39
p_HA = k_HA / n_HA
L_HA = p_HA * n_HA  # lambda

N = 100  # total loop iterations
x = np.arange(N)

# Poisson for null hypothesis
pmf_poisson_H0 = np.zeros(N, dtype=float)
for i in range(0, N):
    pmf_poisson_H0[i] = calculate_poisson_probability(L_H0, i)

# Poisson for null hypothesis
pmf_poisson_HA = np.zeros(N, dtype=float)
for i in range(0, N):
    pmf_poisson_HA[i] = calculate_poisson_probability(L_HA, i)

# Numerical Integration
alpha_index_x = 45  # T value of significance level
# area_under_curve_simpson = simpson(pmf_poisson_H0[:alpha_index_x])  # default dx=1
# print(f"The area under the curve using simpson is {area_under_curve_simpson}")
for index in range(1, N):
    area = simpson(pmf_poisson_H0[:index])
    print(f"index={index}, area={area}")


# Plotting
plt.plot(x, pmf_poisson_H0, label="Poisson H0")
plt.plot(x, pmf_poisson_HA, label="Poisson HA")
plt.plot([alpha_index_x, alpha_index_x], [0, 0.07], color='red', linestyle='dashed', label="alpha")
plt.xlim([0, N])
plt.legend()
plt.grid()
plt.title(f"Null (H0) vs Alternative (HA) Hypothesis | "
          f"H0_mean = {x[np.argmax(pmf_poisson_H0)]} | "
          f"HA_mean = {x[np.argmax(pmf_poisson_HA)]} | ")
plt.xlabel("Trials")
plt.ylabel("Probability")
plt.show()