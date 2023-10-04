import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import trange
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from myLibrary import calculate_sample_cov_matrix


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
# regression model using top-100 selected features

# find 100 features with highest coefficients
model_coef = np.load("data/p2_unsupervised_reduced/model_coefficients.npy")
sum_column = np.abs(model_coef).sum(axis=0)
coef_features = np.argsort(np.abs(sum_column))[::-1][:100]

# load evaluation train and test data
x_train = np.load("data/p2_evaluation_reduced/X_train.npy")
x_test = np.load("data/p2_evaluation_reduced/X_test.npy")
y_train = np.load("data/p2_evaluation_reduced/y_train.npy")
y_test = np.load("data/p2_evaluation_reduced/y_test.npy")

# use selected features
x_train_selection = x_train[:, coef_features]
x_test_selection = x_test[:, coef_features]

# train model
model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr', penalty='l2')
model.fit(x_train_selection, y_train)

y_pred = model.predict(x_test_selection)

print(f"model score train data (coefficient case): {model.score(x_train_selection, y_train)}")
print(f"model score test data (coefficient case): {model.score(x_test_selection, y_test)}\n")

# confusion_matrix(y_test, y_pred)
# print(classification_report(y_test, y_pred))


# -----------------------------------------------------------------
# regression model using random selected features

# # load evaluation train and test data
# x_train = np.load("data/p2_evaluation_reduced/X_train.npy")
# x_test = np.load("data/p2_evaluation_reduced/X_test.npy")
# y_train = np.load("data/p2_evaluation_reduced/y_train.npy")
# y_test = np.load("data/p2_evaluation_reduced/y_test.npy")

# find 100 random features
pool_of_numbers = np.arange(1, 20001)  # Numbers from 1 to 20,000
random_features = np.random.choice(pool_of_numbers, size=100, replace=False)

# use selected features
x_train_selection = x_train[:, random_features]
x_test_selection = x_test[:, random_features]

# train model
model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr', penalty='l2')
model.fit(x_train_selection, y_train)

y_pred = model.predict(x_test_selection)

print(f"model score train data (random case): {model.score(x_train_selection, y_train)}")
print(f"model score test data (random case): {model.score(x_test_selection, y_test)}\n")
pass
# confusion_matrix(y_test, y_pred)
# print(classification_report(y_test, y_pred))


# -----------------------------------------------------------------
# regression model using high-variance selected features

# # load evaluation train and test data
# x_train = np.load("data/p2_evaluation_reduced/X_train.npy")
# x_test = np.load("data/p2_evaluation_reduced/X_test.npy")
# y_train = np.load("data/p2_evaluation_reduced/y_train.npy")
# y_test = np.load("data/p2_evaluation_reduced/y_test.npy")

# find 100 features with highest variance
S = calculate_sample_cov_matrix(x_train)  # covariance matrix
S_diag = np.diag(S)  # features variances
variance_features = np.argsort(S_diag)[::-1][:100]

# use selected features
x_train_selection = x_train[:, variance_features]
x_test_selection = x_test[:, variance_features]

# train model
model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr', penalty='l2')
model.fit(x_train_selection, y_train)

y_pred = model.predict(x_test_selection)

print(f"model score train data (variance case): {model.score(x_train_selection, y_train)}")
print(f"model score test data (variance case): {model.score(x_test_selection, y_test)}\n")

# confusion_matrix(y_test, y_pred)
# print(classification_report(y_test, y_pred))

# plot histograms
plt.hist(S_diag[variance_features], bins=100, label="100 highest variances")
plt.hist(S_diag[coef_features], bins=10, label="100 highest coefficients")
plt.title("Comparison 100 features with highest coefficients vs 100 features with highest variances ")
# plt.xlabel("Number of Components", size=14)
# plt.ylabel("% Variance Explained", size=14)
plt.legend()
plt.show()


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