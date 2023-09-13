import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from numpy import genfromtxt
from myLibrary import calculate_multilinear_OLS, add_intercept


X = genfromtxt('syn_X.csv', delimiter=',')
y = genfromtxt('syn_y.csv', delimiter=',')

X = add_intercept(X)

beta_hat = calculate_multilinear_OLS(X, y)
print(beta_hat)