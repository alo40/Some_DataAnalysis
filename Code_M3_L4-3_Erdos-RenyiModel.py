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

# # Erdos-Renyi graph
# n = 20
# p = 0.14
# G = nx.erdos_renyi_graph(n, p, seed=None, directed=False)
# pos = nx.kamada_kawai_layout(G)
# nx.draw_networkx(G, pos, labels={i: str(i + 1) for i in range(n)})
# plt.show()

# Bibliographic coupling & Cocitation
# A = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
# A = np.array([[0, 0, 1, 0, 0], [0, 0, 0, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
A = np.array([[0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [1, 1, 0, 0, 1], [0, 0, 0, 0, 0], [0, 1, 0, 1, 0]])
G = nx.from_numpy_array(np.array(A), create_using=nx.DiGraph)
B = A.T @ A  # bibliographic coupling
C = A @ A.T  # cocitation
# pos = nx.kamada_kawai_layout(G)
# nx.draw_networkx(G, pos, labels={i: str(i + 1) for i in range(len(A))})
# plt.show()
pass