import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier

from LogisticRegression import LogisticRegression
from DecisionTree import DecisionTree
from LinearRegression import LinearRegression


#Linear Regression
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

# %% Logistic Regression
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

from sklearn.datasets import load_digits

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

X = X[perm[n_test:]]
y = y[perm[n_test:]]

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