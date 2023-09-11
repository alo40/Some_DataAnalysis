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
# number_events = number_events.replace(0, 20)  # only for testing (not valid)
# number_events = number_events.replace(1, 10)  # only for testing (not valid)
average_rate = number_events / time_interval
df['rate'] = average_rate  # only for comparison

# # Testing only
# k = np.arange(0, 100)
# i = 20  # series index
# L_i = average_rate[i] * time_interval[i]
# poisson_distribution = np.exp(-L_i) * (L_i ** k) / factorial(k)

# MLE
sample_size = df.shape[0]
average_rate_estimator = number_events.sum() / (sample_size * time_interval.mean())
# poisson_parameter_estimator = average_rate_estimator * time_interval.mean()
poisson_parameter_estimator = number_events.sum() / sample_size
print(f"average rate estimator = {average_rate_estimator}")
print(f"Poisson parameter estimator = {poisson_parameter_estimator}")

# Estimated Poisson distribution
k = np.arange(0, 100)
poisson_distribution = np.exp(-poisson_parameter_estimator) * (poisson_parameter_estimator ** k) / factorial(k)

# Plotting
ax[0, 0].plot(average_rate, 'x')
ax[0, 0].plot([0, 100], [average_rate_estimator, average_rate_estimator], '--', label='average rate estimator')
ax[0, 0].set_title('average rate (per second) [number of events / time interval]')
ax[0, 0].legend()

ax[0, 1].plot(poisson_distribution, color='red', label='poisson distribution estimator')
ax[0, 1].hist(number_events, bins=np.arange(0, 200), density='True', label='observations')
ax[0, 1].set_title(f"Poisson distribution for paramater = {poisson_parameter_estimator}")
ax[0, 1].set_xlim(0, 100)
ax[0, 1].legend()

ax[1, 0].plot(time_interval, '+')
ax[1, 0].set_title('time interval [seconds] = t_i')

ax[1, 1].plot(number_events, 'v')
ax[1, 1].set_title('number of events [counts] = G_i')

plt.show()


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
