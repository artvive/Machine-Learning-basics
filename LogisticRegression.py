import numpy as np
from utils import *

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def mse(x, y):
    return np.mean((x-y)**2)

def compute_grad(f, a, h):
    dim = a.shape[0]
    grad = np.zeros((dim,1))
    epsilon = np.zeros((dim,1))

    for i in range(dim):
        epsilon[i] = h
        grad[i] = (f(a + epsilon) - f(a - epsilon)) / (2 * h)
        epsilon[i] = 0
    return grad


class LogisticRegression(object):
    """Logistic Regression"""
    def __init__(self):
        super(LogisticRegression, self).__init__()

    def fit(self, X, y):
        X, y = check_transform(X, y)

        self.n_samples = X.shape[0]
        self.n_param = X.shape[1] + 1

        self.design = np.zeros((X.shape[0], X.shape[1] + 1))
        self.design[:, 0] = 1
        self.design[:, 1:] = X

        tol = 1e-6
        tol_it = 10
        epsilon = 1e-3
        step = 1
        self.coef_ = np.random.normal(size=self.n_param, scale = 0.1)
        self.coef_ = self.coef_[:,np.newaxis]
        grad = 1
        i = 0
        j = 0
        momentum = 0

        score = self.score(X, y)
        diff_score = 1
        def objective(x):
            logit = np.matmul(self.design, x)
            return mse(sigmoid(logit), y)

        while abs(diff_score) > tol:
            grad = compute_grad(objective, self.coef_, epsilon)
            new_coef = self.coef_ - grad * step + 0.95*momentum
            momentum = new_coef - self.coef_
            self.coef_ = new_coef

            new_score = objective(self.coef_)
            diff_score = new_score - score
            score = new_score
            if diff_score > 0:
                step = step / 2
                print("New step", step)

            if i%1000 == 0:
                print(diff_score, self.score(X, y))
            i += 1
            self.grad = grad
            #print(np.sqrt(np.sum(grad**2)), step)

        print(i," iterations", grad, step, self.score(X, y))
        self.mse = self.score(X, y)


    def predict(self, X):
        X = check_transform(X)

        intercept = np.ones((X.shape[0], 1))
        new_X = np.concatenate([intercept, X], axis=1)
        #new_coef = self.coef_.reshape(self.coef_.shape[0], 1)
        return sigmoid(np.matmul(new_X, self.coef_))[:, 0]

    def score(self, X, y):
        pred = self.predict(X)
        if pred.shape != y.shape:
            pred = pred.reshape(y.shape)
        return mse(pred, y)



