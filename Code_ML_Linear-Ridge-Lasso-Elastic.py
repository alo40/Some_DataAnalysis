import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from myLibrary import calculate_ResidualSquareSum


# # ------------------------------------------------------------------------------------------------------
# # DATA

# # generate linear noisy 2d data
# np.random.seed(10)
# slope = 4.0
# intercept = 3.0
# x = np.random.uniform(0, 10, 100)
# noise = np.random.normal(0, 2, 100)  # Add random noise with mean 0 and standard deviation 2
# y = slope * x + intercept + noise

# # generate non-linear noisy data
# np.random.seed(10)
# x = np.array([i*np.pi/180 for i in range(60, 300, 4)])
# y = np.sin(x) + np.random.normal(0,0.2,len(x))

# # p2_evaluation_reduced
# x_train = np.load("data/p2_evaluation_reduced/X_train.npy")
# x_test = np.load("data/p2_evaluation_reduced/X_test.npy")
# y_train = np.load("data/p2_evaluation_reduced/y_train.npy")
# y_test = np.load("data/p2_evaluation_reduced/y_train.npy")

# # choosing a cluster as data
# # x = np.load("data/p1/X.npy")
# x = np.load("data/p2_unsupervised_reduced/X.npy")
# x = np.log2(x + 1)
# z = PCA().fit_transform(x)
# c = KMeans(n_clusters=7, n_init=100).fit_predict(x)
# d = np.concatenate((z[:, 0][:, np.newaxis], z[:, 1][:, np.newaxis], c[:, np.newaxis]), axis=1)
# np.save('data/M2_HO/data.npy', d)  # do it only once!

# save data as .npy
data = np.load("data/M2_HO/data.npy")
d = pd.DataFrame(data=data, columns=['x', 'y', 'cluster'])
cluster_1 = 1.0
cluster_2 = 1.0
cluster_3 = 1.0
x = d['x'][(d['cluster'] == cluster_1) | (d['cluster'] == cluster_2) | (d['cluster'] == cluster_3)]
y = d['y'][(d['cluster'] == cluster_1) | (d['cluster'] == cluster_2) | (d['cluster'] == cluster_3)]
x = np.array(x)
y = np.array(y)

# # testing plotting
# plt.scatter(d['x'], d['y'], c=d['cluster'])  # for testing only
# for i, label in enumerate(d['cluster']):
#     plt.annotate(str(label), (d['x'][i], d['y'][i]), fontsize=7)
# plt.show()

# split data
x_train, x_test, y_train, y_test = train_test_split(x[:, np.newaxis], y[:, np.newaxis], test_size=0.2, shuffle=True)


# # ------------------------------------------------------------------------------------------------------
# # REGRESSION MODEL

# LinearRegression from sklearn
linear_reg = LinearRegression().fit(x_train, y_train)
linear_coef = linear_reg.coef_[0]
linear_intercept = linear_reg.intercept_
linear_y = linear_coef * x + linear_intercept
# linear_RSS = calculate_ResidualSquareSum(x_test, y_test, a=linear_coef, b=-1, c=linear_intercept)
# print(f"Linear regression RSS: {linear_RSS}")
# print(f"linear regression score: {linear_reg.score(x_test, y_test)}")

alphas = np.logspace(-4, 4, 1000, endpoint=True)
alphas = [200]

# RidgeCV regression from sklearn (L2)
ridge_reg = RidgeCV(alphas=alphas, cv=10).fit(x_train, y_train)
ridge_coef = ridge_reg.coef_[0]
ridge_intercept = ridge_reg.intercept_
ridge_y = ridge_coef * x + ridge_intercept
# ridge_RSS = calculate_ResidualSquareSum(x_test, y_test, a=ridge_coef, b=-1, c=ridge_intercept)
# print(f"Ridge regression RSS: {ridge_RSS}")
# print(f"ridge regression score: {ridge_reg.score(x_test, y_test)}")
#
# LassoCV regression from sklearn (L1)
lasso_reg = LassoCV(alphas=alphas, cv=10).fit(x_train, y_train)
lasso_coef = lasso_reg.coef_[0]
lasso_intercept = lasso_reg.intercept_
lasso_y = lasso_coef * x + lasso_intercept
# lasso_RSS = calculate_ResidualSquareSum(x_test, y_test, a=lasso_coef, b=-1, c=lasso_intercept)
# print(f"Lasso regression RSS: {lasso_RSS}")
# print(f"lasso regression score: {lasso_reg.score(x_test, y_test)}")

# ElasticNetCV from sklearn (L1+L2)
elastic_reg = ElasticNetCV(alphas=alphas, cv=10, l1_ratio=0.9).fit(x_train, y_train)
elastic_coef = elastic_reg.coef_[0]
elastic_intercept = elastic_reg.intercept_
elastic_y = elastic_coef * x + elastic_intercept
# elastic_RSS = calculate_ResidualSquareSum(x_test, y_test, a=elastic_coef, b=-1, c=elastic_intercept)
# print(f"ElasticNet regression RSS: {elastic_RSS}")
# print(f"ElasticNet regression score: {elastic_reg.score(x_test, y_test)}")


# # ------------------------------------------------------------------------------------------------------
# # PLOTTING

plt.plot(x_train, y_train, 'ro', label='train data')
plt.plot(x_test, y_test, 'bo', label='test data')
plt.plot(x, linear_y, 'r', label='linear regression')
plt.plot(x, ridge_y, 'b', label=f'ridge regression, alpha={ridge_reg.alpha_:.2f}')
plt.plot(x, lasso_y, 'm', label=f'lasso regression, alpha={lasso_reg.alpha_:.2f}')
plt.plot(x, elastic_y, 'c', label=f'elastic regression, alpha={elastic_reg.alpha_:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f"linear score: {linear_reg.score(x_test, y_test):.4f}, "
          f"ridge score: {ridge_reg.score(x_test, y_test):.4f}, "
          f"lasso score: {lasso_reg.score(x_test, y_test):.4f}, "
          f"elastic score: {elastic_reg.score(x_test, y_test):.4f}")
plt.legend()
plt.show()


# # ------------------------------------------------------------------------------------------------------
# # Logistic Regression (not working)
# solver = 'liblinear'  # Options: 'liblinear', 'saga'
# penalty = None  # Options: 'l1', 'l2', 'elasticnet', None
# l1_ratio = None # Options: None, 0.5
# model = LogisticRegression(solver=solver, C=0.05, multi_class='ovr', penalty=penalty, l1_ratio=l1_ratio)
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)


# # ------------------------------------------------------------------------------------------------------
# # load data
# url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
# dataframe = pd.read_csv(url, header=None)
# data = dataframe.values
# x, y = data[:, :-1], data[:, -1]
#
# # cross-validation
# model = Ridge(alpha=1.0)
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# scores = cross_val_score(model, x, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# scores = np.absolute(scores)
# print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


# # ------------------------------------------------------------------------------------------------------
# # Cross-validation scripts
# from yellowbrick.datasets import load_concrete
# from yellowbrick.regressor import AlphaSelection
# from sklearn.linear_model import LassoCV

# # select the best alpha for ridge scores
# alpha_space = np.logspace(0, 2, 100)
# ridge_scores_mean = []
# ridge_scores_std = []
# for alpha in alpha_space:
#     ridge = Ridge(alpha=alpha)
#     ridge_cv_scores = cross_val_score(ridge, x[:, np.newaxis], y[:, np.newaxis], cv=10)
#     ridge_scores_mean.append(np.mean(ridge_cv_scores))
#     ridge_scores_std.append(np.std(ridge_cv_scores))
# plt.plot(alpha_space, ridge_scores_mean, 'ro-', label='mean')  # select inflexion point as alpha
# # plt.plot(alpha_space, ridge_scores_std, 'bx-',label='std')
# plt.xlabel('alpha values')
# plt.ylabel('Ridge score means')
# plt.show()

# # Instantiate the linear model and visualizer
# alphas = np.logspace(-10, 1, 400)
# model = LassoCV(alphas=alphas)
# visualizer = AlphaSelection(model)
# visualizer.fit(x[:, np.newaxis], y[:, np.newaxis])
# # visualizer.show()
# # plt.clf()  # clear figure


# # ------------------------------------------------------------------------------------------------------
# # Not used
# # generate simple 2d data
# x = np.linspace(0, 1, 2)
# y = 1 + x + x * np.random.random(len(x))

# # using direct linear regression
# A = np.vstack([x_train, np.ones(len(x_train))]).T
# alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)), y[:, np.newaxis])

# # calculate Residual Sum of Squares from base line y0 to points
# a_0 = 0
# b_0 = 1
# c_0 = 0
# d_0 = (np.abs(a_0 * x + b_0 * y + c_0) / np.sqrt(a_0 ** 2 + b_0 ** 2)) ** 2
# RSS_0 = d_0.sum()
# print(f"SSR for base line y0: {RSS_0}")

# # calculate Residual Sum of Squares from fitted line to points
# a_1 = alpha[0]
# b_1 = -1
# c_1 = alpha[1]
# d_1 = (np.abs(a_1 * x + b_1 * y + c_1) / np.sqrt(a_1 ** 2 + b_1 ** 2)) ** 2
# RSS_1 = d_1.sum()
# print(f"SSR for fitted line y1: {RSS_1}")

# plt.plot(x_test, y_pred, 'mo', label='sklearn linear regression')
# plt.plot(x, alpha[0] * x + alpha[1], 'b', label='direct linear regression')
# plt.plot(x, y, 'm.')
# plt.plot(x, (a_0 * x + c_0)/-b_0, 'b', label='base line')
# plt.plot(x, (a_1 * x + c_1)/-b_1, 'bx')

# np.save('data/M2_HO/x.npy', df['x'][df['cluster'] == 0])  # do it only once!
# np.save('data/M2_HO/y.npy', df['y'][df['cluster'] == 0])  # do it only once!
# np.save('data/M2_HO/x.npy', df['x'][(df['cluster'] == 0) | (df['cluster'] == 6)])  # do it only once!
# np.save('data/M2_HO/y.npy', df['y'][(df['cluster'] == 0) | (df['cluster'] == 6)])  # do it only once!
