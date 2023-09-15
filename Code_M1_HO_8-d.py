import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from numpy import genfromtxt
from myLibrary import calculate_multilinear_OLS, add_intercept, cal_cost, gradient_descent
import scipy.stats as stats
import statsmodels.api as sm
import pylab


# import zipfile
# import numpy as np
#
# # returns a 3-tuple of (list of city names, list of variable names, numpy record array with each variable as a field)
# def read_mortality_csv(zip_file):
#     import io
#     import csv
#     fields, cities, values = None, [], []
#     with io.TextIOWrapper(zip_file.open('data_and_materials/mortality.csv')) as wrap:
#         csv_reader = csv.reader(wrap, delimiter=',', quotechar='"')
#         fields = next(csv_reader)[1:]
#         for row in csv_reader:
#             cities.append(row[0])
#             values.append(tuple(map(float, row[1:])))
#     dtype = np.dtype([(name, float) for name in fields])
#     return cities, fields, np.array(values, dtype=dtype).view(np.recarray)
#
# with zipfile.ZipFile("release_statsreview_release1.zip") as zip_file:
#     m_cities, m_fields, m_values = read_mortality_csv(zip_file)

csv_file_path = "mortality.csv"
df = pd.read_csv(csv_file_path, index_col=[0])
df = (df-df.mean())/df.std()  # normalization

y = df[['Mortality']].to_numpy()
X = df[df.columns.values[1:]].to_numpy()
X = add_intercept(X)

# num_parameters = X.shape[1]
# theta = np.random.rand(num_parameters, 1)
# theta, cost_history, theta_history, convergence = gradient_descent(X, y, theta, learning_rate=0.0001)
#
#
# # df.plot()
# plt.plot(cost_history)
# plt.yscale("log")
# plt.show()

i = 6
sm.qqplot(df[df.columns.values[i]])
pylab.title(f"{df.columns.values[i]}")
pylab.show()