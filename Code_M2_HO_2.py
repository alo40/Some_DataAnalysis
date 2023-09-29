import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
# import pandas as pd
# import polars as pl
from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score, silhouette_samples
# from yellowbrick.style.colors import resolve_colors

# # transform npy to csv (use only once)
# X = np.load("data/p1/X.npy")
# pd.DataFrame(X).to_csv("data/p1/X.csv")

# # polars (too slow)
# X = pl.read_csv('data/p1/X.csv')

X = np.load("data/p1/X.npy")
X = np.log2(X + 1)

# # k-means colors
# kmeans = KMeans(n_clusters=50, n_init=10)
# y = kmeans.fit_predict(X)

# # PCA
# pca = PCA(n_components=2)
# pca.fit(X)
#
# kmeans = KMeans(n_clusters=4, n_init=10)
# y = kmeans.fit_predict(X)
#
# fig, ax = plt.subplots(1, 1, figsize=(10, 3))
# ax.scatter(X.dot(pca.components_[0]),  X.dot(pca.components_[1]), c=y)
# ax.set(xlabel="Projected data onto first PCA component", ylabel="y")
# plt.tight_layout()
#
# # plt.plot(np.arange(2, 5), [silhouette_score(X, KMeans(i, n_init=50).fit(X).labels_) for i in range(2, 5)])
# plt.show()


# TSNE visualization
pca = PCA()  # Initialize with n_components parameter to only find the top eigenvectors
z = pca.fit_transform(X)
tsne = TSNE(n_components=2, perplexity=140, verbose=1)
X_embedded = tsne.fit_transform(z[:, 0:50])

kmeans = KMeans(n_clusters=50, n_init=10)
y = kmeans.fit_predict(X)

# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y)
# plt.axis("equal")
# plt.show()


# # PCA
# pca = PCA(n_components=2)
# pca.fit(X)
#
# # kmeans = KMeans(n_clusters=4, n_init=10)
# # y = kmeans.fit_predict(X)
#
# fig, ax = plt.subplots(1, 1, figsize=(10, 3))
# ax.scatter(X.dot(pca.components_[0]),  X.dot(pca.components_[1]), c=y)
# ax.set(xlabel="Projected data onto first PCA component", ylabel="y")
# plt.tight_layout()
#
# # plt.plot(np.arange(2, 5), [silhouette_score(X, KMeans(i, n_init=50).fit(X).labels_) for i in range(2, 5)])
# plt.show()


# MDS visualization
embedding = MDS(n_components=2, normalized_stress='auto')
X_transformed = embedding.fit_transform(X)
X_transformed.shape
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y)
plt.show()


# ###########################################################################################
# # not used
# print(f"var cumulative sum = {suma}")
# print(f"index where cumsum > 0.85 is {np.where(suma > 0.85)[0][0]}")

# # elbow plot using k-means
# n = 10
# all_kmeans = [KMeans(n_clusters=i+1, n_init=10) for i in range(n)]
# for i in range(n):
#     all_kmeans[i].fit(X_embedded)
# intertias = [all_kmeans[i].inertia_ for i in range(n)]
# plt.title("KMeans Sum of Squares Criterion")
# plt.xlabel("# cluster")
# plt.ylabel("within-cluster sum of squares")
# plt.plot(intertias)
# plt.xlim(0, n)
# plt.xticks(np.arange(0, n, step=1))
# plt.grid()
# plt.show()

# # PCA
# var = pca.explained_variance_ratio_
# suma = var.cumsum()

# # plotting without visualization technique
# plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label="samples")
# for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
#     comp = comp * var  # scale component by its variance explanation power
#     plt.plot(
#         [0, comp[0]],
#         [0, comp[1]],
#         label=f"Component {i}",
#         linewidth=5,
#         color=f"C{i + 2}",
#     )
# plt.gca().set(
#     aspect="equal",
#     title="2-dimensional dataset with principal components",
#     xlabel="first feature",
#     ylabel="second feature",
# )
# plt.legend()
# plt.show()

# TSNE
# plt.plot(np.arange(2,10),
#          [silhouette_score(X_embedded,KMeans(i,n_init=50).fit(X_embedded).labels_) for i in range(2,10)])
