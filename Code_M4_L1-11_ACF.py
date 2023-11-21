import numpy as np


def calculate_autocovariance(X, h):
    n = X.size
    sum = 0
    for i in range(n - h):
        try:
            sum += (X[i] - X.mean()) * (X[i + h] - X.mean())
        except IndexError:
            print('out of index')
    return 1 / n * sum


# X = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
# X = np.array([-1, 0, 1, 0, -1, 0, 1])
X = np.array([-1, 0, 1, 0, -1, 0, 1, 0])
for i in range(6):
    print(f"i = {i}, autocov = {calculate_autocovariance(X, i)}")
