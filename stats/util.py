import numpy as np

def nansem(x, axis=None):
    x = np.array(x, float)
    return np.nanstd(x, axis=axis) / np.sqrt(np.sum(~np.isnan(x), axis=axis))