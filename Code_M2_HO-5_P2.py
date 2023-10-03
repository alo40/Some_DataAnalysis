import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import pandas as pd
from myLibrary import find_nearest_value, find_nearest_index
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from tqdm import trange



# data load
X = np.load("data/p2_unsupervised/X.npy")
X = np.log2(X + 1)
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()


# PCA
p_components = 1000
n_cluster = 15
pca = PCA(p_components)  # log-transformed data projected on 50 PC's
Z = pca.fit_transform(X)
#
kmeans_type = KMeans(n_clusters=3, n_init=100)
colors_type = kmeans_type.fit_predict(Z)
#
kmeans_subtype = KMeans(n_clusters=n_cluster, n_init=100)
colors_subtype = kmeans_subtype.fit_predict(Z)
#
# labels = np.array_str(np.unique(colors))
# labels = np.unique(colors)
# plt.scatter(Z[:, 0], Z[:, 1])  #, c=colors)
# plt.show()


# TSNE
perplexity = 40
tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity)
z_tsne = tsne.fit_transform(Z)
# plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=colors)
#
# for type in np.unique(colors_type):
#     for subtype in np.unique(colors_subtype):
#
#         # plt.scatter(z_tsne[colors_subtype == subtype, 0],
#         #             z_tsne[colors_subtype == subtype, 1],
#         #             label=f'Type: {type}, Subtype: {subtype}')
#
#         expr1 = colors_type == type
#         expr2 = colors_subtype == subtype
#         element_wise_comparison = expr1 == expr2
#         plt.scatter(z_tsne[element_wise_comparison, 0],
#                     z_tsne[element_wise_comparison, 1],
#                     label=f'Type: {type}, Subtype: {subtype}')
#
for label in np.unique(colors_subtype):
    plt.scatter(z_tsne[colors_subtype == label, 0], z_tsne[colors_subtype == label, 1], label=f'Subtype {label}')
for i, label in enumerate(colors_type):
    plt.annotate(str(label), (z_tsne[i, 0], z_tsne[i, 1]), fontsize=7)
#
plt.title(f"PCA using PC {p_components} ==> TSNE using perplexity {perplexity} | n_cluster = {n_cluster}")
plt.axis("equal")
plt.legend()
plt.show()


# # cumulative variance plot
# pca = PCA()
# Z = pca.fit_transform(X)
# S = np.cumsum(pca.explained_variance_ratio_)
# plt.plot(S)
# plt.title("Cumulative Variance Explained", size=18)
# plt.xlabel("Number of Components", size=14)
# plt.ylabel("% Variance Explained", size=14)
# plt.xscale("log")
# S_index = find_nearest_index(S, 0.80)
# S_value = find_nearest_value(S, 0.80)
# plt.plot([S_index, S_index], [0, S_value], 'r--')
# plt.plot([0, S_index], [S_value, S_value], 'r--')
# plt.grid()
# plt.show()


# # elbow plot using intertia (very slow)
# n = 15  # total number of clusters analyzed
# WGSS = [KMeans(i, n_init=20).fit(X).inertia_ for i in trange(1, n)]
# plt.plot(WGSS)
# plt.title("KMeans Sum of Squares Criterion", size=18)
# plt.xlabel("# Clusters", size=14)
# plt.ylabel("Within-Cluster Sum of Squares", size=14)
# plt.xticks(np.arange(1, n, step=1))
# c = 3  # selected cluster
# plt.plot([c, c], [np.min(WGSS), WGSS[c]], 'r--')
# plt.plot([0, c], [WGSS[c], WGSS[c]], 'r--')
# plt.grid()
# plt.show()
