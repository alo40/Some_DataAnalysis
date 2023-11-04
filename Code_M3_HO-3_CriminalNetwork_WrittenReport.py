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

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

phase = 9
axis = 0  # for subplots
# metric = 'degree'  # 'eigenvector', 'betweenness', 'degree'

pos = nx.spring_layout(G[phase])
for metric in ['eigenvector', 'betweenness', 'degree']:
    match metric:
        case 'eigenvector':
            metric_centrality = nx.eigenvector_centrality(G[phase])
            color = 'cool'  # 'spring'
        case 'betweenness':
            metric_centrality = nx.betweenness_centrality(G[phase])
            color = 'cool'  #'autumn'
        case 'degree':
            metric_centrality = nx.degree_centrality(G[phase])
            color = 'cool'  #'summer'

    # normalization
    min_eigen = min(metric_centrality.values())
    max_eigen = max(metric_centrality.values())
    normalized_metric_centrality = {
        node: (eigen - min_eigen) / (max_eigen - min_eigen)
        for node, eigen in metric_centrality.items()
    }

    # plotting with colormap
    colormap = plt.cm.get_cmap(color)
    node_colors = [colormap(value) for value in normalized_metric_centrality.values()]
    # pos = nx.spring_layout(G[phase])  # Adjust layout algorithm as needed
    nx.draw(G[phase], pos, node_color=node_colors, with_labels=True, ax=axs[axis])

    # adding colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])  # This line is necessary
    cbar = plt.colorbar(sm, orientation='horizontal', label=f"normalized {metric} centrality", ax=axs[axis])

    axs[axis].set_title(f"{metric} centrality")
    axis += 1
fig.suptitle(f"Criminal network phase {phase}", fontsize=16)
plt.show()
