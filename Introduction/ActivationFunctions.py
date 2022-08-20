import numpy as np


def sigmoid(x):              # sigmoid function
    y = 1 / (1 + np.exp(-x))
    return y


def d_sigmoid(y):            # sigmoid derivative function
    return y * (1 - y)
