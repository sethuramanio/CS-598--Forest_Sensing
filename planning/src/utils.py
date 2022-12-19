import numpy as np


def euclidean(p1, p2):
    p1x, p1y = p1
    p2x, p2y = p2
    return np.sqrt((p1x-p2x)**2 + (p1y-p2y)**2)