import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.linalg import eigh


def _graph_from_neigh(neigh):
    n_points = neigh.shape[0]
    graph = lil_matrix((n_points, n_points), dtype='int')
    for i, neighbors in enumerate(neigh):
        for j in neighbors:
            graph[i, j] = 1
            graph[j, i] = 1
    return graph.tocsr()


class Isomap(object):
    """docstring for Isomap"""

    def __init__(self, n_components=2, n_neighbors=10):
        super(Isomap, self).__init__()
        self.n_components = n_components
        self.n_neighbors = n_neighbors

    def fit_transform(self, x):
        n_points = x.shape[0]
        dist = cdist(x, x, metric="sqeuclidean")
        neigh = np.argsort(dist, axis=1)[:, 1:self.n_neighbors + 1]
        graph = _graph_from_neigh(neigh)
        dist = shortest_path(graph)**2
        proj = np.identity(n_points)
        one_vec = np.ones((n_points, 1))
        proj = proj - np.matmul(one_vec, one_vec.transpose()) / n_points
        gram = -0.5 * np.matmul(proj, np.matmul(dist, proj))
        eig_val, eig_vect = eigh(
            gram, eigvals=(n_points - self.n_components, n_points - 1))
        diag = np.sqrt(np.diag(eig_val))
        return np.matmul(eig_vect, diag)
