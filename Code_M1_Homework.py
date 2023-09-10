import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

csv_file_path = "gamma-ray.csv"
df = pd.read_csv(csv_file_path)

df['parameter'] = df['count'] * df['seconds']

fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(df.index.values, df['seconds'], 'v')
ax[0, 0].set_xlabel('index')
ax[0, 0].set_ylabel('seconds')
ax[0, 0].set_ylim(0, 300)

ax[0, 1].plot(df.index.values, df['count'], '+')
ax[0, 1].set_xlabel('index')
ax[0, 1].set_ylabel('count')

ax[1, 0].plot(df.index.values, df['parameter'], 'x')
ax[1, 0].set_xlabel('index')
ax[1, 0].set_ylabel('parameter')
ax[1, 0].set_ylim(0, 300)

ax[1, 1].plot(df['seconds'], df['count'], 'o')
ax[1, 1].set_xlabel('seconds')
ax[1, 1].set_ylabel('count')

plt.show()
