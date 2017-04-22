import numpy as np
from scipy.linalg import eigh


class PCA(object):
    """docstring for PCA"""

    def __init__(self, n_components=2):
        super(PCA, self).__init__()
        self.n_components = n_components

    def fit_transform(self, x):
        dim = x.shape[1]
        x = np.copy(x)
        x = x - np.mean(x, axis=0)
        cov = np.cov(x.transpose())
        val, vect = eigh(cov, eigvals=(dim - self.n_components, dim - 1))
        return np.matmul(x, vect)
