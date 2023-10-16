import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import itertools

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

G = nx.from_numpy_array(A)
modularity = nx.community.modularity(G, [{0, 2, 4, 6, 8}, {1, 3, 5, 7, 9}])
print(f"modularity: {modularity}")
nx.draw_shell(G, with_labels=True, font_weight='bold')
plt.show()
