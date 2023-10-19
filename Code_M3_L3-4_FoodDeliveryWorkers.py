import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# A = np.array([[1, 0, 1, 0, 1],
#               [0, 1, 0, 1, 0],
#               [1, 1, 1, 1, 1],
#               [0, 1, 0, 1, 0],
#               [1, 0, 1, 0, 1]])
# print(f"A * A.T: \n {A @ A.T}")
# print(f"A.T * A: \n {A.T @ A}")


# B = np.array([[1, 0, 1, 0, 1],
#               [0, 1, 0, 1, 0],
#               [1, 1, 1, 1, 1],
#               [0, 1, 0, 1, 0],])

B = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [1, 1, 1, 1],
              [0, 1, 0, 1],
              [1, 0, 1, 0]])

# B = np.array([[1, 0, 1, 0, 1],
#               [0, 1, 0, 1, 0],
#               [1, 1, 1, 1, 1],
#               [0, 1, 0, 1, 0],
#               [1, 0, 1, 0, 1]])

m = B.shape[0]
n = B.shape[1]

# Create a zero matrix of appropriate dimensions
zero_matrix_top = np.zeros((m, m))
zero_matrix_bottom = np.zeros((n, n))

# Assemble matrix A
top_row = np.hstack((zero_matrix_top, B))
bottom_row = np.hstack((B.T, zero_matrix_bottom))
A = np.vstack((top_row, bottom_row))
print(f"A: {A}")

G = nx.from_numpy_array(np.array(A), create_using=nx.Graph)
# pos = nx.bipartite_layout(G)  # not working
pos = nx.kamada_kawai_layout(G)
nx.draw_networkx(G, pos, labels={i: str(i + 1) for i in range(len(A))})
plt.show()


