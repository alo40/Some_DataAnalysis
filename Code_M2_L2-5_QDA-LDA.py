import numpy as np
from numpy.linalg import inv

mu0 = np.array([0, 0])
mu1 = np.array([1, 0])
A = np.array([[1, 0], [0, 1]])

v0 = 1 / np.sqrt(2) * np.array([1,  1])
v1 = 1 / np.sqrt(2) * np.array([1, -1])
v = np.array([v0, v1]).T

sigma = v @ A @ v.T
normal = ((mu0 - mu1).T @ inv(sigma)).T
pass