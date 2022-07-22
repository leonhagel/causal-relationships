import numpy as np
from copy import deepcopy

def basic(X, y, ratio=None, random_state=123):
    """ratio: positive / negative"""
    if not isinstance(X, type(np.array([]))):
        X = np.array(deepcopy(X))
    if not isinstance(y, type(np.array([]))):
        y = np.array(deepcopy(y))
    y = y.reshape(-1)
    if ratio is not None:    
        rng = np.random.default_rng(random_state)
        positive = np.hstack([X[y==1, :], np.ones((X[y==1, :].shape[0], 1))])
        negative = np.hstack([X[y==0, :], np.zeros((X[y==0, :].shape[0], 1))])
        negative_index = rng.choice(range(negative.shape[0]), int(1/ratio*positive.shape[0]), replace=False)
        output = np.vstack([positive, negative[negative_index, :]])
        rng.shuffle(output)
        X, y = output[:, :-1], output[:, -1].reshape(-1, 1)
    else:
        X, y = X, y.reshape(-1, 1)
    return X, y