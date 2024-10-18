'''
Implentation of the KMeans algorithm.
See https://jeremyschroeter.com/2024/04/15/kmeans_gmm.html for more details.
'''

import numpy as np
norm = np.linalg.norm


class KMeans:

    def __init__(self, n_clusters, max_iter=100, init: str = 'km++'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init


    def fit(self, X: np.ndarray) -> np.ndarray:
        
        # initialize values
        if self.init == 'random':
            prev_centers = self._random_init(X)
        elif self.init == 'km++':
            prev_centers = self._kmpp_init(X)
        else:
            raise ValueError(f'"{self.init}" is not a valid init. choose "km++" or "random"')
        
        # initialize labels and compute initial score
        prev_labels = self._assign_labels(X, prev_centers)
        J_scores = [self._J(X, prev_centers, prev_labels)]


        # begin update iterations
        for iteration in range(self.max_iter):

            # optimize center wrt labels
            new_centers = self._update_centers(X, prev_labels)
            J_scores.append(self._J(X, new_centers, prev_labels))

            # optimizer labels wrt centers
            new_labels = self._assign_labels(X, new_centers)
            J_scores.append(self._J(X, new_centers, new_labels))

            # check if fit is good enough
            if (J_scores[-2] - J_scores[-1]) < 1e-8:
                break

            prev_centers, prev_labels = new_centers, new_labels

        self._labels = prev_labels
        self._centroids = prev_centers
        self._cost_trace = J_scores


    def _kmpp_init(self, X: np.ndarray):
        
        init_mus = np.zeros(shape=(self.n_clusters, X.shape[1]))

        for center in range(self.n_clusters):
            
            # choose first point randomly
            if center == 0:
                center_idx = np.random.choice(len(X))
                init_mus[center] = X[center_idx]
            
            else:
                # compute distance from each point to its nearest center
                distances = norm(X[:, np.newaxis] - init_mus, axis=2)
                dist_to_closest_center = distances.min(axis=1)
                
                # choose next center w/ prob proportional to squared dist to nearest center
                dist_to_closest_center **= 2
                probs = dist_to_closest_center / dist_to_closest_center.sum()
                center_idx = np.random.choice(len(X), p=probs)
                init_mus[center] = X[center_idx]

        return init_mus


    def _random_init(self, X: np.ndarray):
        return X[np.random.choice(len(X), self.n_clusters, replace=False)]


    def _J(self, X, mu, labels):
        j = 0
        for k in range(self.n_clusters):
            j += (norm(X[labels == k] - mu[k], axis=1) ** 2).sum()
        return j


    def _assign_labels(self, X: np.ndarray, mu: np.ndarray) -> np.ndarray:
        distances = np.zeros((len(X), self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = norm(X - mu[k], axis=1)
        return distances.argmin(axis=1)


    def _update_centers(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        new_mu = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            new_mu[k] = X[labels == k].mean()
        return new_mu
