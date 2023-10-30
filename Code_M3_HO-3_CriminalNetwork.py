import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time


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

# line_nodes = np.array(range(1, 12+1))
# line_edges = np.array(range(1, 12+1))
# for i in range(1, 12):
phases_list = [9, 10, 11]
for i in phases_list:
  # print(f"phase {i}, eigenvector centrality = {nx.eigenvector_centrality(G[i])}\n")
  # print(f"phase {i}, betweenness centrality = {nx.betweenness_centrality(G[i], normalized=True)}\n")
  # print(f"phase {i}, degree centrality = {nx.degree_centrality(G[i])}\n")
  num_nodes = G[i].number_of_nodes()
  num_edges = G[i].number_of_edges()
  print(f"phase {i}, nodes = {num_nodes}, edges = {num_edges}\n")
  nx.draw_networkx(G[i], ax=axs[i - phases_list[0]])
  axs[i - phases_list[0]].set_title(f"phase {i}")
fig.suptitle("Graphs of criminal network evolution")
plt.show()
  #
#   line_nodes[i] = num_nodes
#   line_edges[i] = num_edges
# plt.plot(line_nodes, 'rx-', label='nodes')
# plt.plot(line_edges, 'mo-', label='edges')
# #
# # seizure
# n = 4
# plt.plot([n, n], [0, 60], 'b--')
# plt.text(n, 10, f'seizure phase {n}', rotation=90, ha='right', va='bottom')
# #
# # seizure
# n = 6
# plt.plot([n, n], [0, 60], 'b--')
# plt.text(n, 10, f'seizure phase {n}', rotation=90, ha='right', va='bottom')
# #
# # seizure
# n = 8
# plt.plot([n, n], [0, 60], 'b--')
# plt.text(n, 10, f'seizure phase {n}', rotation=90, ha='right', va='bottom')
# #
# plt.ylim(0, 60)
# plt.xticks(np.arange(0, 12))
# plt.legend()
# plt.title('Evolution of a criminal network')
# plt.ylabel('number of nodes/edges')
# plt.xlabel('phases')
# plt.show()


# time_eingevector = np.zeros(12)
# time_betweenness = np.zeros(12)
# time_degree = np.zeros(12)
# for i in range(1, 12):
#
#   start = time.time()
#   nx.eigenvector_centrality(G[i])
#   end = time.time()
#   time_eingevector[i] = end - start
#   print(f"phase {i}, eigenvector centrality dT = {end - start}")
#
#   start = time.time()
#   nx.betweenness_centrality(G[i], normalized=True)
#   end = time.time()
#   time_betweenness[i] = end - start
#   print(f"phase {i}, betweenness centrality dT = {end - start}")
#
#   start = time.time()
#   nx.degree_centrality(G[i])
#   end = time.time()
#   time_degree[i] = end - start
#   print(f"phase {i}, degree centrality dT = {end - start}\n")
#
# fig, ax = plt.subplots()
# ax.plot(time_eingevector, label='eigenvector')
# ax.plot(time_betweenness, label='betweenness')
# ax.plot(time_degree, label='degree')
# ax.set_xticks(np.arange(0, 12))
# ax.set_title("Comparison execution time")
# ax.set_xlabel("phases [-]")
# ax.set_ylabel("execution time [s]")
# plt.legend()
# plt.show()


# # list of uniques
# list_uniques = []
# for i in range(1, 12):
#   list_uniques += list(phases[i].columns.values)
# list_uniques = list(set(list_uniques))
#
# results = pd.DataFrame(0.0, index=range(1, 12), columns=list_uniques)
# for i in range(1, 12):
#   for node in list_uniques:
#     # dict = nx.eigenvector_centrality(G[i])
#     dict = nx.betweenness_centrality(G[i], normalized=True)
#     try:
#       results.iloc[i-1][node] = dict[node]
#     except KeyError as e:
#       pass
#     # print(f'I got a KeyError in phase {i} - reason {str(e)}')
# print(results.mean().nlargest(3))
