import numpy as np


def sigmoid(X):
    return 1.0 / (1.0 + np.exp(-X))
