import numpy as np
from numpy.linalg import det
from special import

class MultivariateNormal:

    def __init__(self, mean: np.ndarray, cov: np.ndarray):

        self.mean = mean
        self.cov = cov
        self.k = mean.shape[0]

    def pdf(self):
        Z = (2*np.pi) ** (self.k / 2) * (det(self.cov) ** 2)



