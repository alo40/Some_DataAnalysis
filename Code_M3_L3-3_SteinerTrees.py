import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# # Another example
# A = [[0, 3, 0, 5, 2, 0],
#     [3, 0, 5, 0, 2, 0],
#     [0, 5, 0, 3, 0, 2],
#     [5, 0, 3, 0, 0, 2],
#     [2, 2, 0, 0, 0, 2],
#     [0, 0, 2, 2, 2, 0]]
# G = nx.from_numpy_array(np.array(A), create_using=nx.Graph)
# pos = nx.kamada_kawai_layout(G)
# nx.draw_networkx(G, pos, labels={i: str(i + 1) for i in range(len(A))})
# nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))

# spanning tree
D = [[0, 3, 6, 5],
    [3, 0, 5, 6],
    [6, 5, 0, 3],
    [5, 6, 3, 0]]
GD = nx.from_numpy_array(np.array(D), create_using=nx.Graph)
posD = nx.kamada_kawai_layout(GD)
mst = nx.algorithms.tree.mst.minimum_spanning_tree(GD)
print(mst.size(weight='weight'))
nx.draw_networkx(GD, posD, labels={i: str(i + 1) for i in range(len(GD))})
nx.draw_networkx_edge_labels(GD, posD, edge_labels=nx.get_edge_attributes(GD, 'weight'))
plt.show()
