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
        
        # Initialize values
        if self.init == 'random':
            prev_centers = self._random_init(X)
        elif self.init == 'km++':
            prev_centers = self._kmpp_init(X)
        else:
            raise ValueError(f'"{self.init}" is not a valid init. choose "km++" or "random"')
        
        prev_labels = self._assign_labels(X, prev_centers)

        center_updates = [prev_centers]
        assignment_updates = [prev_labels]
        J_scores = [self._J(X, prev_centers, prev_labels)]


        # Begin update iterations
        for iteration in range(self.max_iter):
            new_centers = self._update_centers(X, prev_labels)
            J_scores.append(self._J(X, new_centers, prev_labels))

            new_labels = self._assign_labels(X, new_centers)
            J_scores.append(self._J(X, new_centers, new_labels))

            if (J_scores[-1] - J_scores[-2]) < 1e-4:
                break

            center_updates.append(new_centers)
            assignment_updates.append(new_labels)

            prev_centers, prev_labels = new_centers, new_labels

        self._labels = prev_labels
        self._centroids = prev_centers
        self._cost_trace = J_scores
        self._centroid_updates = center_updates
        self._label_updates = assignment_updates


    def _kmpp_init(self, X: np.ndarray):
        NotImplemented
    
    def _random_init(self, X: np.ndarray):
        return np.random.choice(len(X), self.n_clusters, replace=False)



    def _J(self, X, means, labels):
        NotImplemented

    def _assign_labels(self, X: np.ndarray, means: np.ndarray) -> np.ndarray:
        NotImplemented

    def _update_centers(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        NotImplemented
