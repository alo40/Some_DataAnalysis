import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


# data load
X = np.load("data/p2_unsupervised/X.npy")
X = np.log2(X + 1)
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()


# PCA
p_components = 1000
n_cluster = 12
pca = PCA(p_components)  # log-transformed data projected on 50 PC's
Z = pca.fit_transform(X)
kmeans = KMeans(n_clusters=n_cluster, n_init=100)
colors = kmeans.fit_predict(Z)
# plt.scatter(Z[:, 0], Z[:, 1])  #, c=colors)
# plt.show()


# TSNE
perplexity = 40
tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity)
z_tsne = tsne.fit_transform(Z)
plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=colors)
plt.title(f"PCA using PC {p_components} ==> TSNE using perplexity {perplexity} | n_cluster = {n_cluster}")
plt.axis("equal")
plt.show()


# # cumulative variance plot
# plt.plot(np.arange(0, 2169), np.cumsum(pca.explained_variance_ratio_))
# plt.title("Cumulative Variance Explained", size=18)
# plt.xlabel("Number of Components", size=14)
# plt.ylabel("% Variance Explained", size=14)
# plt.xscale("log")
# plt.grid()
# plt.show()


# # elbow plot using intertia (option 1)
# n = 20
# plt.plot(np.arange(1, n), [KMeans(i, n_init=50).fit(X).inertia_ for i in range(1, n)])
# plt.title("KMeans Sum of Squares Criterion", size=18)
# plt.xlabel("# Clusters", size=14)
# plt.ylabel("Within-Cluster Sum of Squares", size=14)
# plt.xticks(np.arange(1, n, step=1))
# plt.grid()
# plt.show()
