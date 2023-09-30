import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.manifold import TSNE


# # log-transform
X = np.load("data/p1/X.npy")
X = np.log2(X + 1)
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()
kmeans = KMeans(n_clusters=5, n_init=100)
colors = kmeans.fit_predict(X)


# # PCA
# pca = PCA(50)  # log-transformed data projected on 50 PC's
# Z = pca.fit_transform(X)
# # kmeans = KMeans(n_clusters=5, n_init=100)
# # colors = kmeans.fit_predict(Z)
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


# # mean value of data points for each cluster (not used!)
# df = pd.DataFrame(X)
# df['Colors'] = colors
# X_means = pd.DataFrame()
# for i in range(1, 5):
#     X_means[f'{i}'] = df[df['Colors'] == i].mean(axis=1)


###########################################################################################
# CLUSTER CHECKS
###########################################################################################

# # elbow plot
# plt.plot(np.arange(0, 11), pca.explained_variance_ratio_[0:11])
# plt.title("% Explained variance by component", size=18)
# plt.xlabel("Component #", size=14)
# plt.ylabel("% Variance Explained", size=14)
# plt.xticks(np.arange(0, 11, step=1))
# plt.grid()
# plt.show()


# # elbow plot using intertia (option 1)
# n = 5
# plt.plot(np.arange(1, n), [KMeans(i, n_init=50).fit(X).inertia_ for i in range(1, n)])
# plt.title("KMeans Sum of Squares Criterion", size=18)
# plt.xlabel("# Clusters", size=14)
# plt.ylabel("Within-Cluster Sum of Squares", size=14)
# plt.xticks(np.arange(1, n, step=1))
# plt.grid()
# plt.show()


# # elbow plot using intertia (option 2)
# all_kmeans = [KMeans(n_clusters=i+1,n_init=100) for i in range(8)]
# # i-th kmeans fits i+1 clusters
# for i in range(8):
#     all_kmeans[i].fit(X)
# inertias = [all_kmeans[i].inertia_ for i in range(8)]
# plt.plot(np.arange(1,9),inertias)
# plt.title("KMeans Sum of Squares Criterion",size=18)
# plt.xlabel("# Clusters",size=14)
# plt.ylabel("Within-Cluster Sum of Squares",size=14)
# plt.show()


# # cumulative variance plot
# plt.plot(np.arange(0, 511), np.cumsum(pca.explained_variance_ratio_))
# plt.title("Cumulative Variance Explained", size=18)
# plt.xlabel("Number of Components", size=14)
# plt.ylabel("% Variance Explained", size=14)
# plt.xscale("log")
# plt.grid()
# plt.show()
