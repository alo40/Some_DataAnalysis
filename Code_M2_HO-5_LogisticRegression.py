import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import trange
from sklearn.manifold import TSNE


# # -----------------------------------------------------------------
# # example 2 parameters

# x = np.arange(10).reshape(-1, 1)
# y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
#
# model = LogisticRegression(solver='liblinear', random_state=0)
# model.fit(x, y)
# model.predict(x)
# model.score(x, y)

# confusion_matrix(y, model.predict(x))
# print(classification_report(y, model.predict(x)))


# # -----------------------------------------------------------------
# # example Multiparameters

# x, y = load_digits(return_X_y=True)
#
# x_train, x_test, y_train, y_test =\
#     train_test_split(x, y, test_size=0.2, random_state=0)
#
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
#
# model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr', random_state=0)
# model.fit(x_train, y_train)
#
# x_test = scaler.transform(x_test)
# y_pred = model.predict(x_test)
#
# model.score(x_train, y_train)
# model.score(x_test, y_test)
#
# confusion_matrix(y_test, y_pred)
# print(classification_report(y_test, y_pred))


# -----------------------------------------------------------------
# calculate regression model

# # load original data
# X = np.load("data/p2_unsupervised_reduced/X.npy")
# X = np.log2(X + 1)
# x_train = X
# n_cluster = 9
# y_train = KMeans(n_clusters=n_cluster, n_init=100).fit_predict(X)
# # np.save('data/p2_unsupervised_reduced/x_train.npy', x_train)  # do it only once!
# # np.save('data/p2_unsupervised_reduced/y_train.npy', y_train)  # do it only once!

# # load train data
# x = np.load("data/p2_unsupervised_reduced/x_train.npy")
# y = np.load("data/p2_unsupervised_reduced/y_train.npy")
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
#
# # train model
# model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr', penalty='l2')
# model.fit(x_train, y_train)
#
# y_pred = model.predict(x_test)
#
# model.score(x_train, y_train)
# model.score(x_test, y_test)
#
# confusion_matrix(y_test, y_pred)
# print(classification_report(y_test, y_pred))

# # np.save('data/p2_unsupervised_reduced/model_coefficients.npy', model.coef_)  # do it only once!


# -----------------------------------------------------------------
# regression model using selected features

# find top 100 features
model_coef = np.load("data/p2_unsupervised_reduced/model_coefficients.npy")
sum_column = np.abs(model_coef).sum(axis=0)
top_features = np.argsort(np.abs(sum_column))[::-1][:100]

# load evaluation train and test data
x_train = np.load("data/p2_evaluation_reduced/X_train.npy")
x_test = np.load("data/p2_evaluation_reduced/X_test.npy")
y_train = np.load("data/p2_evaluation_reduced/y_train.npy")
y_test = np.load("data/p2_evaluation_reduced/y_test.npy")

# use selected features
x_train_selection = x_train[:, top_features]
x_test_selection = x_test[:, top_features]

# train model
model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr', penalty='l2')
model.fit(x_train_selection, y_train)

y_pred = model.predict(x_test_selection)

model.score(x_train_selection, y_train)
model.score(x_test_selection, y_test)

# confusion_matrix(y_test, y_pred)
# print(classification_report(y_test, y_pred))


# # -----------------------------------------------------------------
# # regression model using selected features
#
# # load evaluation train and test data
# x_train = np.load("data/p2_evaluation_reduced/X_train.npy")
# x_test = np.load("data/p2_evaluation_reduced/X_test.npy")
# y_train = np.load("data/p2_evaluation_reduced/y_train.npy")
# y_test = np.load("data/p2_evaluation_reduced/y_test.npy")



# # -----------------------------------------------------------------
# # elbow methdd
#
# n = 15  # total number of clusters analyzed
# WGSS = [KMeans(i, n_init=50).fit(X).inertia_ for i in trange(1, n)]
# plt.plot(WGSS)
# plt.title("KMeans Sum of Squares Criterion", size=18)
# plt.xlabel("# Clusters", size=14)
# plt.ylabel("Within-Cluster Sum of Squares", size=14)
# plt.xticks(np.arange(1, n, step=1))
# plt.grid()
# plt.show()


# # -----------------------------------------------------------------
# # TSNE
#
# perplexity = 40
# tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity)
# z_tsne = tsne.fit_transform(X)
# for label in np.unique(y_train):
#     plt.scatter(z_tsne[y_train == label, 0], z_tsne[y_train == label, 1], label=f'Subtype {label}')
# #
# plt.title(f"TSNE using perplexity {perplexity} | n_cluster = {n_cluster}")
# plt.axis("equal")
# plt.legend()
# plt.show()