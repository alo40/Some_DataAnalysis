import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


features, target = load_wine(return_X_y=True)
features.shape
wine_std = (features-np.mean(features,0))/np.std(features,0)

# PCA
# pca_wine = PCA(5).fit(wine_std)
# pcs = pca_wine.transform(wine_std)

X = np.load("data/p1/X.npy")
X = np.log2(X + 1)
pca = PCA(2).fit(X)
pcs = pca.transform(X)

# plt.scatter(pcs[:,0],pcs[:,1])
# plt.title("Wine Data PCs",size=18)
# plt.xlabel("PC 1",size=14)
# plt.ylabel("PC 2",size=14)
# plt.axis("equal")
# plt.show()
#
# # MDS
# mds_wine = MDS(2).fit_transform(wine_std)
# plt.scatter(mds_wine[:,0],mds_wine[:,1])
# plt.title("Wine Data MDS",size=18)
# plt.axis("equal")
# plt.show()

# TSNE perpexitly loop
# for perplexity in [5,10,30,50,80,100]:
#     tsne_wine = TSNE(n_components=2,perplexity=perplexity).fit_transform(wine_std)
#     plt.scatter(tsne_wine[:,0],tsne_wine[:,1])
#     plt.title("Wine Data TSNE, perplexity "+str(perplexity),size=18)
#     plt.axis("equal")
#     plt.show()

# # TSNE
# tsne_wine = TSNE(n_components=2,perplexity=50).fit_transform(wine_std)
# plt.scatter(tsne_wine[:,0],tsne_wine[:,1])
# plt.title("Wine Data TSNE, perplexity 50",size=18)
# plt.axis("equal")
# plt.show()

# # elbow plot
# plt.plot(np.arange(1,6),[KMeans(i,n_init=50).fit(wine_std).inertia_ for i in range(1,6)])
# plt.title("KMeans Sum of Squares Criterion",size=18)
# plt.xlabel("# Clusters",size=14)
# plt.ylabel("Within-Cluster Sum of Squares",size=14)
# plt.show()

# # silhouette plot
# plt.plot(np.arange(2,6),[silhouette_score(wine_std,KMeans(i,n_init=50).fit(wine_std).labels_) for i in range(2,6)])
# plt.title("Average Silhouette Scores",size=18)
# plt.xlabel("# Clusters",size=14)
# plt.ylabel("Average Silhouette Score",size=14)
# plt.show()


from sklearn.metrics import silhouette_score, silhouette_samples
from yellowbrick.style.colors import resolve_colors
from yellowbrick.cluster import SilhouetteVisualizer

# Horizontal axes on the plots are not all the same range, be careful
# for i in range(2, 6):

i = 2
clustering = KMeans(i, n_init=50)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
visualizer = SilhouetteVisualizer(clustering, colors='yellowbrick', is_fitted=False, ax=ax1)
visualizer.fit(X)

colors = np.array(resolve_colors(i, "yellowbrick"))
ax2.scatter(pcs[:, 0], pcs[:, 1], c=colors[clustering.labels_])
ax2.axis("equal")

# If we want to set axes to be the same for all plots, need to do something like this
# instead of visualizer.show(), which resets the axes
visualizer.finalize()
ax1.set_xlim((-.2, .6))
plt.show()
