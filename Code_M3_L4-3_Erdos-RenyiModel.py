import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.special import comb
from myLibrary import calculate_binomial_probability

# # first exercises
# n = 1
# k = 1
# p = 0.2
# # print(comb(n, k))
# # print(n * (n - 1) / 2)
# print(calculate_binomial_probability(n, k, p))

# Erdos-Renyi graph
n = 20
p = 0.14
G = nx.erdos_renyi_graph(n, p, seed=None, directed=False)
pos = nx.kamada_kawai_layout(G)
nx.draw_networkx(G, pos, labels={i: str(i + 1) for i in range(n)})
plt.show()