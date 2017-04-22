import numpy as np


class GaussianMixture(object):
    """GaussianMixture Implementation"""

    def __init__(self, k, init="random"):
        super(GaussianMixture, self).__init__()
        self.k = k

    def fit(self, x, max_iter=1000):
        mu, sigma, pi = self.init(x)
        p = self.compute_post(x, mu, sigma, pi)
        tol = 1e-8
        p_old = -np.ones((x.shape[0], self.k))
        i = 0
        while np.mean((p - p_old)**2) > tol and i < max_iter:
            i += 1
            pi, mu, sigma = self.compute_params(p, x)
            p_old = p
            p = self.compute_post(x, mu, sigma, pi)
        return mu, sigma, p

    def compute_params(self, p, x):
        dim = x.shape[1]
        mu = p.reshape(-1, self.k, 1) * x.reshape(-1, 1, dim)
        mu = np.sum(mu, axis=0)
        mu = mu / np.transpose(np.sum(p, axis=0, keepdims=True))

        sigma = x.reshape(1, -1, dim) - mu.reshape(-1, 1, dim)
        sigma = np.matmul(sigma.reshape(self.k, -1, dim, 1),
                          sigma.reshape(self.k, -1, 1, dim))
        sigma = p.transpose((1, 0)).reshape(self.k, -1, 1, 1) * sigma
        sigma = np.sum(sigma, axis=1)
        sigma = sigma / np.sum(p, axis=0).reshape(self.k, 1, 1)

        pi = np.mean(p, axis=0)
        return pi, mu, sigma

    def compute_post(self, x, mu, sigma, pi):
        dim = x.shape[1]
        diff = x.reshape(1, -1, dim) - mu.reshape(-1, 1, dim)

        sigm_inv = np.linalg.inv(sigma)
        p = np.matmul(sigm_inv.reshape(self.k, 1, dim, dim),
                      diff.reshape(self.k, -1, dim, 1))
        p = np.matmul(diff.reshape(self.k, -1, 1, dim),
                      p).reshape(self.k, x.shape[0])
        p = p.transpose((1, 0))

        p = np.exp(-p / 2)
        p = p / np.sqrt(np.linalg.det(sigma)).reshape(1, self.k)
        p = p * pi.reshape(1, self.k)
        p = p / np.sum(p, axis=1).reshape(-1, 1)
        return p

    def init(self, x):
        indices = np.random.choice(x.shape[0], size=self.k, replace=False)
        mu = x[indices]
        dim = x.shape[1]
        sigma = np.identity(dim)
        sigma = np.repeat(sigma.reshape(1, dim, dim), self.k, axis=0)
        pi = np.ones(self.k) / self.k
        return mu, sigma, pi
