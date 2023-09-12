import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

csv_file_path = "golub.csv"
df = pd.read_csv(csv_file_path, index_col=[0])

csv_file_path = "golub_cl.csv"
# lukemia_type = pd.read_csv(csv_file_path, index_col=[0])  # not used

# treatment type
df_ALL = df.iloc[:, 0:27]
df_AML = df.iloc[:, 27:38]

# parameters
gen_i = 3051  # not used if for loop used
N_ALL = df_ALL.shape[1]
N_AML = df_AML.shape[1]


p_value_array = np.zeros(df.shape[0])
for gen_i in range(1, df.shape[0]):

    sample_mean_ALL_gen_i = df_ALL.loc[gen_i].mean()
    sample_mean_AML_gen_i = df_AML.loc[gen_i].mean()
    observation_ALL_gen_i = df_ALL.loc[gen_i]
    observation_AML_gen_i = df_AML.loc[gen_i]

    # for ALL
    sample_var_ALL_gen_i = np.sum((observation_ALL_gen_i - sample_mean_ALL_gen_i)**2) / (N_ALL - 1)
    variance_X_ALL_gen_i = sample_var_ALL_gen_i / N_ALL

    # for AML
    sample_var_AML_gen_i = np.sum((observation_AML_gen_i - sample_mean_AML_gen_i)**2) / (N_AML - 1)
    variance_X_AML_gen_i = sample_var_AML_gen_i / N_AML

    variance_metric_diff_gen_i = variance_X_ALL_gen_i + variance_X_AML_gen_i

    # t-test statistic
    t_Welch_gen_i = (sample_mean_ALL_gen_i - sample_mean_AML_gen_i) / np.sqrt(variance_metric_diff_gen_i)

    dof_denominator_ALL = 1 / (N_ALL - 1) * variance_X_ALL_gen_i ** 2
    dof_denominator_AML = 1 / (N_AML - 1) * variance_X_AML_gen_i ** 2
    degrees_freedom_gen_i = variance_metric_diff_gen_i ** 2 / (dof_denominator_ALL + dof_denominator_AML)

    p_value_gen_i = 2 * stats.t.sf(abs(t_Welch_gen_i), degrees_freedom_gen_i)  # two-sided test, as H0 assume same means

    p_value_array[gen_i] = p_value_gen_i

uncorrected_significant_counts = (p_value_array[1:] <= 0.05).sum()
print(f"The number of significant genes for the uncorrected case is {uncorrected_significant_counts}")

m = df.shape[0]
i = np.arange(1, m)
sorted_p_value_array = np.sort(p_value_array[1:])

HB_significant_counts = (sorted_p_value_array * (m - i + 1) <= 0.05).sum()
print(f"The number of significant genes using the Holm-Bonferroni correction is {HB_significant_counts}")

BH_significant_counts = (sorted_p_value_array * m / i <= 0.05).sum()
print(f"The number of significant genes using the Benjamini-Hochberg correction is {BH_significant_counts}")

plt.plot(0.05 / (m - i + 1), '-r', label="Holm-Bonferroni correction")
plt.plot(0.05 * i / m, '-b', label="Benjamini-Hochberg correction")
plt.plot(np.sort(p_value_array[1:]), '-g', label="sorted p-values")  # don't count index zero
# plt.xlim(0, 120)
# plt.ylim(0, 0.00002)
plt.yscale("log")
plt.legend()
plt.show()

# # my results
# print("\nMy results:")
# print(f"t_statistic = {t_Welch_gen_i}")
# print(f"degrees of freedom = {degrees_freedom_gen_i}")
# print(f"p-value for gen {gen_i} = {p_value_gen_i}")
#
# # verification (maybe wrong)
# statistic_gen_i = stats.ttest_ind(observation_ALL_gen_i, observation_AML_gen_i, equal_var=False)
# print("\nVerification:")
# print(f"{statistic_gen_i}\n")
#
#
# fig, ax = plt.subplots(1, 1)
#
# # per gen
# ax.plot(df_ALL.loc[gen_i], 'bx', label='ALL')
# ax.plot(df_AML.loc[gen_i], 'r+', label='AML')
# ax.set_title(f"golub_data for gen {gen_i}")
# ax.legend()
# ax.grid()
#
# gen_2 = 101
# ax[1].plot(df_ALL.loc[gen_2], '-bx', label='ALL')
# ax[1].plot(df_AML.loc[gen_2], '-r+', label='AML')
# ax[1].set_title(f"golub_data for gen {gen_2}")
# ax[1].legend()

# # per patient
# patient_1 = "V9"
# patient_2 = "V25"
# ax[0].plot(df[f"{patient_1}"], 'bx')
# ax[1].plot(df[f"{patient_2}"], 'ro')

# plt.show()

# # means (not correct!)
# plt.plot(df_ALL.mean(), label='ALL', marker='x')
# plt.plot(df_AML.mean(), label='AML', marker='o')