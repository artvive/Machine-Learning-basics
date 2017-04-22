import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import eigh


class LocallyLinearEmbedding(object):
    """docstring for LocallyLinearEmbedding"""

    def __init__(self, n_components=2, n_neighbors=10):
        super(LocallyLinearEmbedding, self).__init__()
        self.n_components = n_components
        self.n_neighbors = n_neighbors

    def fit_transform(self, x):
        n_points = x.shape[0]
        w = np.zeros((n_points, n_points))
        dist = cdist(x, x, metric="sqeuclidean")
        neigh = np.argsort(dist, axis=1)[:, 1:self.n_neighbors + 1]
        one = np.ones(self.n_neighbors)
        for i in range(n_points):
            gram = x[neigh[i]]
            gram = gram - x[i]
            gram = np.matmul(gram, gram.transpose())
            gram = gram + 0.001 * np.identity(self.n_neighbors)
            w_perm = np.linalg.solve(gram, one)
            w[i, neigh[i]] = w_perm
            w[i] = w[i] / np.sum(w[i])
        m = np.identity(n_points) - w
        m = np.matmul(m.transpose(), m)
        eig_val, eig_vect = eigh(
            m, eigvals=(0, self.n_components))
        return eig_vect[:, 1:]
