import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import itertools

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

A = np.array([[0,1,1,1],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0]])

graph = nx.from_numpy_array(np.array(A), create_using=nx.DiGraph)
# eigenvector_centrality = nx.eigenvector_centrality(graph)
# print(eigenvector_centrality)
katz_centrality = nx.katz_centrality(graph)
print(katz_centrality)
nx.draw_shell(graph, with_labels=True, font_weight='bold')
plt.show()


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
