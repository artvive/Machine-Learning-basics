import numpy as np
from scipy.spatial.distance import cdist


class KMeans(object):
    """KMeans Implementation"""

    def __init__(self, k, init="++"):
        super(KMeans, self).__init__()
        self.k = k
        self.init = init

    def fit(self, x):
        if self.init == "random":
            means = self.init_random(x)
        if self.init == "++":
            means = self.init_plus(x)

        classes, dist = self.assign(x, means)
        old_classes = -np.ones(x.shape[0])
        hist = [np.sum(dist)]

        while np.any(old_classes != classes):
            means = self.compute_means(x, classes)
            old_classes = np.copy(classes)
            classes, dist = self.assign(x, means)
            hist = hist + [np.sum(dist)]

        hist = np.array(hist)
        return means, classes, hist

    def init_random(self, x):
        indices = np.random.choice(x.shape[0], size=self.k, replace=False)
        return x[indices]

    def init_plus(self, x):
        dim = x.shape[1]
        means = x[np.random.randint(x.shape[0])]
        means = means.reshape(1, dim)
        for i in range(1, self.k):
            classes, dist = self.assign(x, means)
            prob = dist / np.sum(dist)
            new_index = np.random.choice(x.shape[0], p=prob)
            means = np.concatenate((means, x[new_index].reshape(1, dim)),
                                   axis=0)
        return means

    def assign(self, x, means):
        dist = self.compute_dists(x, means)
        classes = np.argmin(dist, axis=1)
        dist = dist[(range(x.shape[0]), classes)]
        return classes, dist

    def compute_dists(self, x, means):
        dist = cdist(x, means, metric="sqeuclidean")
        return dist

    def compute_means(self, x, classes):
        dim = x.shape[1]
        means = np.zeros((self.k, dim))
        for i in range(self.k):
            means[i] = np.mean(x[classes == i], axis=0)
        return means
