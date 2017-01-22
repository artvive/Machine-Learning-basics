import numpy as np
from utils import *


class LinearRegression(object):
    """
    Linear Regression:
    
    """
    def __init__(self):
        super(LinearRegression, self).__init__()

    def fit(self, X, y):
        self.n_samples = X.shape[0]
        self.n_param = X.shape[1] + 1

        self.design = np.zeros((X.shape[0], X.shape[1] + 1))
        self.design[:, 0] = 1
        self.design[:, 1:] = X
        sym_design = np.matmul(self.design.T, self.design)
        transformed_y = np.matmul(self.design.T, y[np.newaxis, :].T)
        self.coef = np.linalg.solve(sym_design, transformed_y).T

    def predict(self, X):
        intercept = np.ones((X.shape[0], 1))
        new_X = np.concatenate([intercept, X], axis=1)
# new_coef = self.coef_.reshape(self.coef_.shape[0], 1)
        return np.matmul(new_X, self.coef.T)[:, 0]

    def score(self, X, y):
        pred = self.predict(X)
        if pred.shape != y.shape:
            pred = pred.reshape(y.shape)
        return mse(pred, y)
