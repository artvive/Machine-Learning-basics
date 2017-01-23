#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 12:41:26 2016

@author: arthur
"""

import numpy as np
from sklearn.metrics import log_loss


def entropy(x):
    vals = np.unique(x)
    n_samples = x.shape[0]
    probs = np.array([np.sum(x == i) / n_samples for i in vals])
    return np.sum(probs * np.log(probs))


def score_split(X, y, value):
    mask = X <= value
    left_y = y[mask]
    right_y = y[np.logical_not(mask)]
    weighted_entropy = left_y.shape[0] * entropy(left_y) + \
                       right_y.shape[0] * entropy(right_y)
    return weighted_entropy / y.shape[0]


def most_probable_value(x):
    vals = np.unique(x)
    n_samples = x.shape[0]
    probs = np.array([np.sum(x == i) / n_samples for i in vals])
    return vals[np.argmax(probs)]


class DecisionTree(object):
    """Decision Tree"""
    def __init__(self):
        super(DecisionTree, self).__init__()
        self.min_sample_leaf = 1
        self.min_samples_split = 2

    def fit(self, X, y):
        self.tree = self.trainTree(X, y)

    def trainTree(self, X, y):
        n_samples = X.shape[0]
        n_feat = X.shape[1]
        ent = entropy(y)
        if n_samples < self.min_samples_split or ent >= 0:
            return ["Leaf", most_probable_value(y), ent]

        best = ent
        best_feat = -1
        best_val = -1

        for i in range(n_feat):
            x_temp = np.unique(X[:, i])
            for j in range(x_temp.shape[0] - 1):
                val = (x_temp[j] + x_temp[j + 1]) / 2
                score = score_split(X[:, i], y, val)
                if score > best:
                    best = score
                    best_val = val
                    best_feat = i

        mask = X[:, best_feat] <= best_val
        left_y = y[mask]
        right_y = y[np.logical_not(mask)]
        left_X = X[mask, :]
        right_X = X[np.logical_not(mask), :]

        return [[best_feat, best_val],
                self.trainTree(left_X, left_y),
                self.trainTree(right_X, right_y)]

    def predict(self, X):
        n_samples = X.shape[0]
        res = np.zeros(n_samples)
        for i in range(n_samples):
            res[i] = self.predictTree(X[i, :], self.tree)
        return res

    def predictTree(self, x, tree):
        if tree[0] == "Leaf":
            return tree[1]
        feat = tree[0][0]
        val = tree[0][1]
        if x[feat] <= val:
            return self.predictTree(x, tree[1])
        else:
            return self.predictTree(x, tree[2])

    def score(self, X, y):
        y_pred = self.predict(X)
        labels = np.unique(y)
        y_prob = np.zeros((y.shape[0], labels.shape[0]))
        for i in range(labels.shape[0]):
            y_prob[:, i] = y_pred == labels[i]
        return np.mean(y_pred == y), log_loss(y, y_prob)

    def capacity(self):
        return capacity(self.tree)


def capacity(tree):
    if tree[0] == "Leaf":
            return 1
    return 1 + capacity(tree[1]) + capacity(tree[2])


if __name__ == "__main__":
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
    from sklearn.tree import DecisionTreeClassifier
    clf2 = DecisionTreeClassifier(criterion="entropy")
    clf2.fit(X, y)
    print("Train score (sklearn): ", clf2.score(X, y),
          log_loss(y, clf2.predict_proba(X)))
    print("Train score (sklearn): ", clf2.score(X_test, y_test),
          log_loss(y_test, clf2.predict_proba(X_test)))
