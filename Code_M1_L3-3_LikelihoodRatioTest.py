from myLibrary import calculate_binomial_probability
import numpy as np


# parameters
p_H0_MLE = (39 + 63) / (31_000 + 31_000)  #
p_HA_MLE_T = 39 / 31_000  # Treatment under HA
p_HA_MLE_C = 63 / 31_000  # Control under HA

L_H0 = calculate_binomial_probability(31_000, 39, p_H0_MLE) * calculate_binomial_probability(31_000, 63, p_H0_MLE)
L_HA = calculate_binomial_probability(31_000, 39, p_HA_MLE_T) * calculate_binomial_probability(31_000, 63, p_HA_MLE_C)

T = -2 * np.log(L_H0 / L_HA)
print(f"The value of the Test Statistic is equal to {T}")
