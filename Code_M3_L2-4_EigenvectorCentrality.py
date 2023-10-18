import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import itertools
import pandas as pd

# A = np.array([[1,0,0,0],
#     [1,0,0,0],
#     [1,0,0,0],
#     [1,0,0,0]])

# A = np.array([[1,1,1,1],
#     [0,0,0,0],
#     [0,0,0,0],
#     [0,0,0,0]])

# A = np.array([[1,1,1,1],
#     [1,0,0,0],
#     [1,0,0,0],
#     [1,0,0,0]])

# A = np.array([[0,1,1,1],
#     [0,0,0,0],
#     [0,0,0,0],
#     [0,0,0,0]])

# A = np.array([[0, 1, 1, 1, 1, 1, 1],
#     [1, 0, 1, 0, 0, 0, 0],
#     [1, 1, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 1, 0, 0],
#     [1, 0, 0, 1, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 1, 0]])


# # Circle of n nodes
# max_nodes = 100
# C_array = np.zeros([max_nodes + 1])
# for num_nodes in range(3, max_nodes + 1):
#     A = np.zeros((num_nodes, num_nodes), dtype=int)
#     for i in range(num_nodes):
#         A[i, (i + 1) % num_nodes] = 1  # Connect to the next node
#         A[i, (i - 1) % num_nodes] = 1  # Connect to the previous node
#     G = nx.from_numpy_array(np.array(A))
#     C = nx.closeness_centrality(G)[0]
#     C_array[num_nodes] = C
#     print(num_nodes)
#     print(f"closeness: {C}")
# # nx.draw_shell(G, with_labels=True, font_weight='bold')
# x = np.arange(1, max_nodes + 2, dtype=float)
# plt.plot(x, C_array, 'r.', label='networkx')
# plt.plot(x+3, x**(-0.6), 'b.', label='theory')
# plt.title('Closeness acc. to number of nodes')
# plt.xlabel('number of nodes')
# plt.ylabel('closeness coefficient')
# plt.legend()
# plt.show()


# Line of n nodes
max_nodes = 8
B_array = np.zeros([max_nodes + 1])
for num_nodes in range(3, max_nodes + 1):
    A = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        if i > 0:
            A[i][i - 1] = 1  # Connect to the previous node
        if i < num_nodes - 1:
            A[i][i + 1] = 1  # Connect to the next node
    G = nx.from_numpy_array(np.array(A))
    B = nx.betweenness_centrality(G)
    # B_array[num_nodes] = B
    print(num_nodes)
    print(f"betweenness: {B}")
# nx.draw_shell(G, with_labels=True, font_weight='bold')
# x = np.arange(1, max_nodes + 2, dtype=float)
# plt.plot(C_array, 'r.', label='networkx')
# plt.plot(x+3, x**(-0.6), 'b.', label='theory')
# plt.title('Closeness acc. to number of nodes')
# plt.xlabel('number of nodes')
# plt.ylabel('closeness coefficient')
# plt.legend()
# plt.show()


# G = nx.from_numpy_array(np.array(A), create_using=nx.DiGraph)
# eigenvector_centrality = nx.eigenvector_centrality(G)
# katz_centrality = nx.katz_centrality(G)
# pagerank_centrality = nx.pagerank(G)
# hubs, authorities = nx.hits(G)
# closeness_centrality = nx.closeness_centrality(G)
# betweenness_centrality = nx.betweenness_centrality(G)
# centrality = pd.DataFrame({'eigenvector': eigenvector_centrality,
#                            'katz': katz_centrality,
#                            'page-rank': pagerank_centrality,
#                            'hubs': hubs,
#                            'authorities': authorities,
#                            'closeness': closeness_centrality,
#                            'betweenness': betweenness_centrality})
# print(centrality)
# nx.draw_shell(G, with_labels=True, font_weight='bold')
# plt.show()


# # A = np.array([[1, 1], [-2, -2]])
# # A = np.array([[2, 1], [-2, -1]])
# A = np.array([[3, 4, -2], [1, 4, -1], [2, 6, -1]])
# eigenvalue_1, eigenvector_1 = np.linalg.eig(A)
# eigenvalue_2, eigenvector_2 = np.linalg.eig(A @ A)
# eigenvalue_3, eigenvector_3 = np.linalg.eig(A @ A @ A)


# G = nx.from_numpy_array(A)
# # modularity = nx.community.modularity(G, [{0, 2, 4, 6, 8}, {1, 3, 5, 7, 9}])
# # print(f"modularity: {modularity}")
#
# # eigenvalues, left_eigenvectors = np.linalg.eig(A.T)
# eigenvalues, right_eigenvectors = np.linalg.eig(A)
# print(eigenvalues)
# print(nx.eigenvector_centrality(G))
