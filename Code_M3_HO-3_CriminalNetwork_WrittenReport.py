import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# read data
phases = {}
G = {}
for i in range(1, 12):
  var_name = "phase" + str(i)
  file_name = "https://raw.githubusercontent.com/ragini30/Networks-Homework/main/" + var_name + ".csv"
  phases[i] = pd.read_csv(file_name, index_col = ["players"])
  phases[i].columns = "n" + phases[i].columns
  phases[i].index = phases[i].columns
  phases[i][phases[i] > 0] = 1
  G[i] = nx.from_pandas_adjacency(phases[i])
  G[i].name = var_name

fig, ax = plt.subplots(1, 1, figsize=(15, 5))

phase = 6
metric = 'degree'  # 'eigenvector', 'betweenness', 'degree'
match metric:
    case 'eigenvector': metric_centrality = nx.eigenvector_centrality(G[phase])
    case 'betweenness': metric_centrality = nx.betweenness_centrality(G[phase])
    case 'degree'     : metric_centrality = nx.degree_centrality(G[phase])

# normalization
min_eigen = min(metric_centrality.values())
max_eigen = max(metric_centrality.values())
normalized_eigenvector_centrality = {
    node: (eigen - min_eigen) / (max_eigen - min_eigen)
    for node, eigen in metric_centrality.items()
}

# plotting with colormap
colormap = plt.cm.get_cmap('cool')
node_colors = [colormap(value) for value in normalized_eigenvector_centrality.values()]
pos = nx.spring_layout(G[phase])  # Adjust layout algorithm as needed
nx.draw(G[phase], pos, node_color=node_colors, with_labels=True, ax=ax)

# adding colorbar
sm = plt.cm.ScalarMappable(cmap=colormap, norm=Normalize(vmin=0, vmax=1))
sm.set_array([])  # This line is necessary
# cbar = plt.colorbar(sm, orientation='vertical', label='Eigenvector Centrality', cax=plt.gcf().add_axes([0.85, 0.15, 0.03, 0.5]))
cbar = plt.colorbar(sm, orientation='vertical', label=f"normalized {metric} centrality")

plt.title(f"{metric} centrality, phase {phase}")  # not working
plt.show()
