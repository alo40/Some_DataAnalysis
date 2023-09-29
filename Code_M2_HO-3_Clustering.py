import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.manifold import TSNE


# # log-transform
X = np.load("data/p1/X.npy")
X = np.log2(X + 1)
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()


# PCA
pca = PCA(50)  # log-transformed data projected on 50 PC's
Z = pca.fit_transform(X)
kmeans = KMeans(n_clusters=5, n_init=100)
colors = kmeans.fit_predict(Z)
# plt.scatter(Z[:, 0], Z[:, 1], c=colors)
# plt.show()


# # MDS
# mds = MDS(n_components=2, verbose=1, eps=1e-5)
# mds.fit(X)
# plt.scatter(mds.embedding_[:, 0], mds.embedding_[:, 1], c=colors)
# plt.show()


# TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40)
z_tsne = tsne.fit_transform(X)
plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=colors)
plt.title("TSNE, perplexity 40", size=18)
plt.axis("equal")
plt.show()


# # elbow plot
# plt.plot(np.arange(0, 11), pca.explained_variance_ratio_[0:11])
# plt.title("% Explained variance by component", size=18)
# plt.xlabel("Component #", size=14)
# plt.ylabel("% Variance Explained", size=14)
# plt.xticks(np.arange(0, 11, step=1))
# plt.grid()
# plt.show()


# # cumulative variance plot
# plt.plot(np.arange(0, 511), np.cumsum(pca.explained_variance_ratio_))
# plt.title("Cumulative Variance Explained", size=18)
# plt.xlabel("Number of Components", size=14)
# plt.ylabel("% Variance Explained", size=14)
# plt.xscale("log")
# plt.grid()
# plt.show()
