import numpy as np
import scipy


def notears(M):
    return np.trace(scipy.linalg.expm(M*M)) - M.shape[0]

def get_thresholds(W, verbose=0):
    thresholds = []
    h_old = 1
    for threshold in np.arange(0, 1, 0.001):
        W_t = np.where(W>threshold, W, 0)
        h = notears(W_t)
        if h < h_old:
            thresholds += [threshold]
        h_old = h
        if verbose >=1:
            print(f"{threshold:.3f}: \t{h}")
        if h < 1e-16:
            break
    return thresholds
