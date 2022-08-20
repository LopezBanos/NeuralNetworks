import numpy as np


def l2_loss(pred, target):
    """
    Common Loss function for supervised learning method.
    :param pred: Array with values predicted by NN.
    :param target: Array with the real values.
    :return: The L2 loss function.
    """
    return np.sum(np.square(pred - target))


def d_l2_loss(pred, target):
    """
    Derivative of L2 loss Function
    :param pred: Array with values predicted by NN.
    :param target: Array with the real values.
    :return:
    """
    return 2 * (pred - target)
