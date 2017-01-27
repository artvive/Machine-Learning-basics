import numpy as np
from utils import *


class LinearRegression(object):
    """
    Linear Regression:
    - TODO: QR decomposition to improve numerical stability
    """
    def __init__(self):
        super(LinearRegression, self).__init__()

    def fit(self, X, y):
        self.n_param = X.shape[1] + 1

        design = np.zeros((X.shape[0], X.shape[1] + 1))
        design[:, 0] = 1
        design[:, 1:] = X

        symmetric_design = np.matmul(design.T, design)
        transformed_y = np.matmul(design.T, y[np.newaxis, :].T)
        self.coef = np.linalg.solve(symmetric_design, transformed_y).T

    def predict(self, X):
        intercept = np.ones((X.shape[0], 1))
        new_X = np.concatenate([intercept, X], axis=1)

        return np.matmul(new_X, self.coef.T)[:, 0]

    def score(self, X, y):
        pred = self.predict(X)
        if pred.shape != y.shape:
            pred = pred.reshape(y.shape)
        return mse(pred, y)
