import numpy as np


def check_transform(X, y=None):
    n_samples = X.shape[0]
    new_X = X
    if len(X.shape) < 2:
        new_X = X[:, np.newaxis]
    if len(X.shape) > 2:
        new_X = X.reshape((n_samples, X[0, :].size))

    if y is None:
        return new_X
    new_y = y
    if n_samples != y.shape[0]:
        raise NotImplementedError

    if len(y.shape) < 2:
        new_y = y[:, np.newaxis]

    if new_y.shape[1] != 1:
        raise NotImplementedError
    return new_X, new_y
