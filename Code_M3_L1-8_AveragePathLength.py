import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import itertools


# Clustering coefficient
# cluster_array = np.array([])
for num_nodes in range(3, 10):
    edges = []
    for edge in itertools.combinations(range(1, num_nodes + 1), 2): edges.append(edge)
    # edges.remove((1, 3))
    G = nx.Graph()
    G.add_edges_from(edges)
    cluster_coef = nx.average_clustering(G)
    print(f"Clustering coefficient (auto): {cluster_coef:.4}, nodes: {num_nodes}")
    # cluster_array = np.append(cluster_array, cluster_coef)

    # Clustering coefficient (manual)
    A = nx.adjacency_matrix(G)
    A = A.toarray()
    A3 = A @ A @ A
    triplets_closed = A3.trace()
    triplets_connected = 0
    for node in G.nodes:
        k = G.degree(node)
        # print(f"node {node} has degree {k}")
        triplets_connected += k * (k - 1)
    print(f"Clustering coefficient (manual): {triplets_closed / triplets_connected}")

# nx.draw_shell(G, with_labels=True, font_weight='bold')
# plt.plot(cluster_array, 'r-')
# plt.show()




# # constrain: node have at most degree 2 and from a tree figure
# # edges = [(0, 1), (0, 2), (1, 3)]
# # edges = [(0, 1), (0, 2), (1, 3), (1, 4)]
# edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
# G = nx.Graph()
# G.add_edges_from(edges)
# shortest_path = dict(nx.shortest_path_length(G))
# node_path_avg = [sum(paths.values()) / len(G.nodes) for node, paths in shortest_path.items()]
# # print(f"node path average: {node_path_avg}")
# nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
# plt.show()


# # constrain: node have at most degree 2 and form a closed figure
# edges = [(1, 2), (2, 3)]
# averages = np.array([])
# m = 100
# for i in range(3, m):
#     edges.pop()
#     edges.append((i-1, i))
#     edges.append((i, 1))
#     n = len(edges)
#     G = nx.Graph()
#     G.add_edges_from(edges)
#     shortest_path = dict(nx.shortest_path_length(G))
#     node_path_avg = [sum(paths.values()) / len(G.nodes) for node, paths in shortest_path.items()]
#     print(f"node path average: {node_path_avg[0]:.4f}, with n: {n}")
#     # print(edges)
#     averages = np.append(averages, node_path_avg[0])
# x = np.arange(3, m)
# plt.plot(x, averages, 'bx')
# y = x/4
# plt.plot(x, y, 'r')
# plt.grid()
# plt.show()


# # star graph
# edges = []
# for i in range(1, 100):
#     edges.append((0, i))
#     G = nx.Graph()
#     G.add_edges_from(edges)
#     shortest_path = dict(nx.shortest_path_length(G))
#     node_path_avg = [sum(paths.values()) / len(G.nodes) for node, paths in shortest_path.items()]
#     print(f"node path average: {node_path_avg[0]:.4f}, {node_path_avg[1]:.4f}")
