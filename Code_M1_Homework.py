import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.special import factorial
from scipy.stats import poisson
from myLibrary import calculate_poisson_probability

csv_file_path = "gamma-ray.csv"
df = pd.read_csv(csv_file_path)


# poisson for average rate
fig, ax = plt.subplots(2, 2)
time_interval = df['seconds']
number_events = df['count']
average_rate = number_events / time_interval
df['rate'] = average_rate  # only for comparison

k = np.arange(0, 100)
i = 20  # series index
L_i = average_rate[i] * time_interval[i]
poisson_distribution = np.exp(-L_i) * (L_i ** k) / factorial(k)
# poisson = calculate_poisson_probability(average_number_events, k)

# MLE
sample_size = df.shape[0]
average_rate_estimator = number_events.sum() / (sample_size * time_interval.mean())
print(f"average rate estimator = {average_rate_estimator}")

ax[0, 0].plot(average_rate, 'x')
ax[0, 0].plot([0, 100], [average_rate_estimator, average_rate_estimator], '--')
ax[0, 0].set_title('average rate (per second) [number of events / time interval]')

ax[0, 1].plot(poisson_distribution, color='red')
ax[0, 1].set_title(f"Poisson distribution for paramater = {L_i}")
# ax[0, 1].set_xlim(0, 10)

ax[1, 0].plot(time_interval, '+')
ax[1, 0].set_title('time interval [seconds] = t_i')

ax[1, 1].plot(number_events, 'v')
ax[1, 1].set_title('number of events [counts] = G_i')

# ax.plot(k, poisson_distribution)
plt.show()


# # observations
# fig, ax = plt.subplots(1, 1)
# df['parameter'] = df['count'] * df['seconds']
# ax.hist(df['parameter'], bins=np.arange(0, 200), density='True', label='observations')
# ax.set_ylim(0, 0.04)
# # ax.set_xlim(50, 150)
#
# # poisson distribution
# lambda_hat = 1 / df.shape[0] * (df['parameter']).sum()  # MLE
# # lambda_hat = 100  # test only
# x = np.arange(0, 500)
# ax.plot(x, poisson.pmf(x, lambda_hat), label='poisson pmf')
#
# plt.legend()
# plt.show()


# # data visualization
# fig, ax = plt.subplots(2, 2)
#
# ax[0, 0].plot(df.index.values, df['seconds'], 'v')
# ax[0, 0].set_xlabel('index')
# ax[0, 0].set_ylabel('seconds')
# ax[0, 0].set_ylim(0, 300)
#
# ax[0, 1].plot(df.index.values, df['count'], '+')
# ax[0, 1].set_xlabel('index')
# ax[0, 1].set_ylabel('count')
#
# ax[1, 0].plot(df.index.values, df['parameter'], 'x')
# ax[1, 0].set_xlabel('index')
# ax[1, 0].set_ylabel('parameter')
# ax[1, 0].set_ylim(0, 300)
#
# ax[1, 1].plot(df['seconds'], df['count'], 'o')
# ax[1, 1].set_xlabel('seconds')
# ax[1, 1].set_ylabel('count')
#
# plt.show()

# # NOT USED
# number_events = number_events.replace(1, 0)  # only for testing (not valid)
# number_events = number_events.replace(2, 0)  # only for testing (not valid)
# number_events = number_events.replace(3, 0)  # only for testing (not valid)
# number_events = number_events.replace(4, 0)  # only for testing (not valid)
# number_events = number_events.replace(5, 0)  # only for testing (not valid)
# number_events = number_events.replace(6, 0)  # only for testing (not valid)
# number_events = number_events.replace(7, 0)  # only for testing (not valid)
# average_number_events = number_events / time_interval  # usual parameter lambda (probably wrong)
