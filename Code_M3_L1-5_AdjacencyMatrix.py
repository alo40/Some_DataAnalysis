import numpy as np
from scipy.sparse.csgraph import connected_components

# # example from https://www.youtube.com/watch?v=MNoXLskj93k
# A = np.array([[0, 1, 1, 1, 1],
#               [1, 0, 0, 0, 0],
#               [1, 0, 0, 0, 0],
#               [1, 0, 0, 0, 0],
#               [1, 0, 0, 0, 0]])
# # print(A)
# # print(A @ A)
# # print(A @ A @ A)
# # print(A @ A @ A @ A)
# # print(A @ A @ A @ A @ A)
# print(A @ A @ A @ A @ A @ A)


# # question 5 from course
# A = np.array([[1, 1, 0],
#               [0, 0, 1],
#               [1, 0, 0]])
# # print(A @ A)
# print(A @ A @ A)

# question 6 from course
A = np.array([[0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
              [1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
              [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 1, 0, 1, 0, 1, 0]])
# print((A == A.T).all())
# print(A @ A @ A @ A @ A)
# print(A @ A)
print(connected_components(A))


# import networkx as nx  # You can use the NetworkX library for graph operations
#
# # Create a graph (you can replace this with your specific graph creation method)
# G = nx.Graph(A)
#
# # Add nodes and edges to your graph (replace this with your graph's structure)
# # G.add_nodes_from([1, 2, 3, 4, 5])
# # G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (2, 4)])
#
# # Initialize a variable to keep track of the maximum degree
# max_degree = -1
#
# # Iterate through all nodes and calculate their degree
# for node in G.nodes():
#     degree = G.degree(node)
#     if degree > max_degree:
#         max_degree = degree
#
# print("Maximum degree in the graph:", max_degree)
