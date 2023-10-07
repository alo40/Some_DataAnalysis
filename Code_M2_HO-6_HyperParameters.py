import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

# load data
# data_path = "data/p1/X.npy"
data_path = "data/p2_unsupervised/X.npy"
X = np.load(data_path)
X = np.log2(X + 1)

# # PCA/TSNE single
# Z_pca = PCA(n_components=10).fit_transform(X)
# Z_tsne = TSNE(n_components=2, perplexity=40).fit_transform(Z_pca)
# plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1])
# plt.show()

# PCA/TSNE multiple
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
pbar = tqdm(total=60)
n_components = [10, 50, 100, 250, 500, 1000]
index = 0
for i in range(0, 2):
    for j in range(0, 3):
        Z_pca = PCA(n_components[index]).fit_transform(X)
        Z_tsne = TSNE(n_components=2, perplexity=40).fit_transform(Z_pca)
        axs[i, j].scatter(Z_tsne[:, 0], Z_tsne[:, 1], c='r')
        axs[i, j].set_title(f'number of PCs used for TSNE plot: {n_components[index]}')
        axs[i, j].set_xlim(-50, 50)
        axs[i, j].set_ylim(-50, 50)
        index += 1
        pbar.update(10)
pbar.close()
plt.suptitle(f"Comparison TSNE using data from {data_path}")
plt.tight_layout()
plt.show()
