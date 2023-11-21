import numpy as np


def calculate_population_variance(X):
    Xmean = 0.5
    n = X.size
    sum = 0
    for i in range(n):
        sum += (X[i] - Xmean) ** 2
    return sum / n


X = np.array([1, -1, -1, -1, 1, 1, -1, -1, -1])
print(calculate_population_variance(X))
