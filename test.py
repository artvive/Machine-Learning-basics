import scipy.stats as st
import numpy as np
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier

from LogisticRegression import LogisticRegression
from DecisionTree import DecisionTree
from LinearRegression import LinearRegression
from GaussianMixture import GaussianMixture
from K_means import KMeans
from Isomap import Isomap
from PCA import PCA
from LLE import LocallyLinearEmbedding

from sklearn.datasets import load_digits, load_iris

import matplotlib.pyplot as plt

# Linear Regression
print("LINEAR REGRESSION TESTS")

temp = np.ones((2, 4))

N = 100

X1 = np.linspace(0, 100, num=N)
X2 = np.logspace(-1, 2, num=N)
X1 = X1[:, np.newaxis]
X2 = X2[:, np.newaxis]

X = np.concatenate([X1, X2], axis=1)
noise = st.norm.rvs(0, 1, size=N)
noise = noise[:, np.newaxis]

y = 86 * X1 - 7 * X2 + noise
y = y.reshape(N)

clf = LinearRegression()

clf.fit(X, y)


y_pred = clf.predict(X)

print("Regression coefs: ", clf.coef)
print("Predictions: ", y_pred[1:10])
print("True values: ", y[1:10])

# Logistic Regression
print("\nLOGISTIC REGRESSION TESTS")
N = 100

X = np.linspace(0, 100, num=N)
y = np.concatenate([np.ones(N // 2), np.zeros(N // 2)])
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / std

clf = LogisticRegression()

clf.fit(X, y)

test_x = np.array([1, 2, 49, 50, 51, 99])

print("Predicted values: ", clf.predict((test_x - mean) / std))
print("True Values: ", y[test_x])

# Decision Tree tests

print("\nDECISION TREE TESTS")
N = 100

X = np.linspace(0, 100, num=N)
X = X[:, np.newaxis]
y = np.concatenate([np.ones(N // 2), np.zeros(N // 2)])
clf = DecisionTree()
clf.fit(X, y)
print("(Accuracy, log loss)")
print("Score on toy data set: ", clf.score(X, y))

print("Mnist dataset:")
df = load_digits()

X = df['data']
y = df['target']
N = y.shape[0]
np.random.seed(123)
perm = np.random.permutation(N)

n_test = N // 10
X_test = X[perm[:n_test]]
y_test = y[perm[:n_test]]

n_train = 1000
X = X[perm[n_test:n_test + n_train]]
y = y[perm[n_test:n_test + n_train]]

clf = DecisionTree()
clf.fit(X, y)

print("Train score (our implementation): ", clf.score(X, y))
print("Test score (our implementation): ", clf.score(X_test, y_test))

clf2 = DecisionTreeClassifier(criterion="entropy")
clf2.fit(X, y)
print("Train score (sklearn): ", (clf2.score(X, y),
      log_loss(y, clf2.predict_proba(X))))
print("Train score (sklearn): ", (clf2.score(X_test, y_test),
      log_loss(y_test, clf2.predict_proba(X_test))))

print("\n MIXTURE OF GAUSSIAN TESTS")


def plot_classes(x, means, classes):
    plt.scatter(x[:, 0], x[:, 1], c=classes)
    plt.scatter(means[:, 0], means[:, 1], c='r', s=30)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


x, classes = load_iris(True)

gm = GaussianMixture(3)
mu, sigma, p = gm.fit(x)
classes = np.argmax(p, axis=1)
plot_classes(x, mu, classes)
print(mu)
print("\n K_MEANS TESTS")

x, classes = load_iris(True)
km = KMeans(3)
means, classes, hist = km.fit(x)
plot_classes(x, means, classes)
print(means)

print("\n PCA TESTS")

x = df['data']
y = df['target']
n_train = 1000
perm = np.random.permutation(x.shape[0])
x = x[perm[:n_train]]
y = y[perm[:n_train]]

pca = PCA(n_components=2)
x_transf = pca.fit_transform(x)
plt.scatter(x_transf[:, 0], x_transf[:, 1], c=y)
plt.show()

print("\n ISOMAP TESTS")

iso = Isomap(n_components=2, n_neighbors=10)
x_transf = iso.fit_transform(x)
plt.scatter(x_transf[:, 0], x_transf[:, 1], c=y)
plt.show()

print("\n LLE TESTS")

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
x_transf = lle.fit_transform(x)
plt.scatter(x_transf[:, 0], x_transf[:, 1], c=y)
plt.show()
