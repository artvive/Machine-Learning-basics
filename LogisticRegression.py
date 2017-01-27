import numpy as np
from utils import check_transform


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def mse(x, y):
    return np.mean((x - y)**2)


class LogisticRegression(object):
    """
    Logistic Regression
    """
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

        step = 1
        self.coef_ = np.random.normal(size=self.n_param, scale=0.1)
        self.coef_ = self.coef_[:, np.newaxis]
        grad = 1
        i = 0
        momentum = 0

        score = self.score(X, y)
        diff_score = 1

        def objective(x):
            logit = np.matmul(self.design, x)
            return mse(sigmoid(logit), y)

        while abs(diff_score) > tol:

            logit = np.matmul(self.design, self.coef_)
            sigm = sigmoid(logit)
            grad = np.sum((y - sigm) * sigm * (1 - sigm) * self.design, axis=0)
            new_coef = self.coef_ - grad * step + 0.95 * momentum
            momentum = new_coef - self.coef_
            self.coef_ = new_coef

            new_score = objective(self.coef_)
            diff_score = new_score - score
            score = new_score
            if diff_score > 0:
                step = step / 2

            i += 1
            self.grad = grad

        self.mse = self.score(X, y)

    def predict(self, X):
        X = check_transform(X)

        intercept = np.ones((X.shape[0], 1))
        new_X = np.concatenate([intercept, X], axis=1)
        return sigmoid(np.matmul(new_X, self.coef_))[:, 0]

    def score(self, X, y):
        pred = self.predict(X)
        if pred.shape != y.shape:
            pred = pred.reshape(y.shape)
        return mse(pred, y)
