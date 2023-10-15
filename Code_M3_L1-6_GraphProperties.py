import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom, norm


# load edges
file_path = "data/release_directed_graph.txt"
E = pd.read_csv(file_path, sep=' ', header=None, names=['start', 'end'])

# create vertex
vertex = E['start'].unique()
status = np.ones(len(vertex), dtype=int) * -1  # used for DFS algorithm, -1: not visited, 0: visited, 1: popped
edges = np.zeros(len(vertex), dtype=int)
V = pd.DataFrame({'vertex': vertex, 'status': status})

# create stack
stack = pd.DataFrame(columns=['stack', 'parent'])
stack.loc[0] = [0, '-']

# count number of edges per vertex
edges = np.array([], dtype=int)
for vertex in V['vertex']:
    # print(E[E['start'] == vertex].shape[0])
    edges = np.append(edges, E[E['start'] == vertex].shape[0])
V['edges'] = edges
V['probability'] = V['edges'] / 100

# under null hypotesis
p = 0.1
n = 100 * 100
x = np.arange(binom.ppf(0.001, n, p), binom.ppf(0.999, n, p))
rv = binom.pmf(x, n, p)
s = binom.std(n, p, loc=0)
mu = binom.mean(n, p, loc=0)
# plt.plot(rv)
# plt.show()

MLE = V['edges'].sum() / (100 * V['edges'].size)

# Z-Test
Z = abs((1030 - mu) / s)

# alternative calculation
s_1 = np.sqrt(p * (1 - p)) / np.sqrt(n)
Z_1 = abs((0.103 - 0.1) / s_1)

# p-value (in a standard normal distribution)
cdf_neg = norm.cdf(-Z, loc=0, scale=1)
cdf_pos = 1 - norm.cdf( Z, loc=0, scale=1)
p_value = cdf_neg + cdf_pos
print(f"p-value is: {p_value}")

# # ----------------------------------------------------------------------------------
# # NOT USED
# # ----------------------------------------------------------------------------------

# # Z-Test (fail!)
# n = V['probability'].size
# sample_mean = V['probability'].mean()
# null_mean = 0.1
# null_std = np.sqrt(n * null_mean * (1 - null_mean)) / np.sqrt(n)
# Z = abs(sample_mean - null_mean / s)

# # DFS algorithm
# next_vertex_index = 0
# while True:
#     next_vertex = E['next'].loc[next_vertex_index]
#     next_vertex_index = E[E['present'] == next_vertex].index[0]
#     pass

# # counting self-loops
# for i in range(E.shape[0]):
#     if E[i, 0] == E[i, 1]:
#         print(f"loop in row {i}")

# # counting directed cycles (Depth First Search)
# stack_index = 1
# for index in range(E.shape[0]):

    # next_vertex = E['head'].loc[index]

    # next_vertex_index = V[V['vertex'] == next_vertex]['status'].index[0]
    # if V.loc[next_vertex_index].values[1] == -1:
    #     V.loc[next_vertex_index].values[1] = 0
    #
    # # E[E['tail'] == vertex].iloc[0, 0]
    #
    # stack.loc[stack_index] = [E['head'].loc[index], E['tail'].loc[index]]
    # stack_index += 1

    # print(E[E['vertex'] == vertex])


# E = np.loadtxt(file_path, dtype=int)
# vertex = np.unique(E[:, 0])
# V = np.column_stack((vertex, status))