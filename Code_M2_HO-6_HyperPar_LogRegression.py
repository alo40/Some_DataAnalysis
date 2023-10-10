import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


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
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

# load M2_HO data
# data = np.load("data/M2_HO/data.npy")
# d = pd.DataFrame(data=data, columns=['x', 'y', 'cluster'])
# cluster_1 = 0.0
# cluster_2 = 1.0
# x = d[['x', 'y']][(d['cluster'] == cluster_1) | (d['cluster'] == cluster_2)]
# y = d['cluster'][(d['cluster'] == cluster_1) | (d['cluster'] == cluster_2)]
# x = np.array(x)
# y = np.array(y)

# load M2_HO train and test data
x1_train = np.load("data/M2_HO/x_train.npy")
x1_test = np.load("data/M2_HO/x_test.npy")
x2_train = np.load("data/M2_HO/y_train.npy")
x2_test = np.load("data/M2_HO/y_test.npy")
y_train = np.load("data/M2_HO/c_train.npy")
y_test = np.load("data/M2_HO/c_test.npy")
x_train = np.concatenate((x1_train, x2_train), axis=1)
x_test = np.concatenate((x1_test, x2_test), axis=1)

# train model
solver = 'saga'  # Options: 'liblinear', 'saga'
penalty = 'elasticnet'  # Options: 'l1', 'l2', 'elasticnet'
l1_ratio = 0.9 # Options: None, 0.5
model = LogisticRegression(solver=solver, C=1/1e10, multi_class='ovr', penalty=penalty, l1_ratio=l1_ratio)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(f"using solver {solver}, "
      f"penalty {penalty} and "
      f"l1_ratio {l1_ratio} the train score is: {model.score(x_train, y_train)}")
print(f"using solver {solver}, "
      f"penalty {penalty} and "
      f"l1_ratio {l1_ratio} the test score is: {model.score(x_test, y_test)}")

# confusion_matrix(y_test, y_pred)
# print(classification_report(y_test, y_pred))

# np.save('data/p2_unsupervised_reduced/model_coefficients.npy', model.coef_)  # do it only once!