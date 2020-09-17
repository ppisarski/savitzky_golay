"""
 Implements the method described in
 https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
 free to use at the user's risk
"""

__all__ = ["savgol_nonuniform"]

import numpy as np


def savgol_nonuniform(x, y, window_length, polyorder, deriv=0):
    """
    Savitzky-Golay smoothing 1D filter

    :param x:
    :param y:
    :param window_length: the smoothing sample, e.g. window_length=2 for smoothing over 5 points
    :param polyorder: the degree of the local polynomial fit, e.g. polyorder=2 for a parabolic fit
    :param deriv: The order of the derivative to compute. This must be a nonnegative integer.
            The default is 0, which means to filter the data without differentiating.
    :return:
    """
    if type(x) is not np.array:
        x = np.array(x)
    if type(y) is not np.array:
        y = np.array(y)

    n = int((window_length - 1) / 2)

    if x.shape != y.shape:
        raise RuntimeError("x and y arrays are of different shape")
    if x.shape[0] < window_length:
        raise RuntimeError("not enough data to start the smoothing process")
    if 2 * n + 1 <= polyorder + 1:
        raise RuntimeError("need at least deg+1 points to make the polynomial")

    # smooth start and end data
    ysm = np.zeros(y.shape)
    for i in range(n):
        j = y.shape[0] - i - 1
        if deriv == 0:
            ysm[i] = y[i]
            ysm[j] = y[j]
        if deriv == 1:
            ysm[i] = (y[i] - y[i - 1]) / (x[i] - x[i - 1])
            ysm[j] = (y[j] - y[j - 1]) / (x[j] - x[j - 1])
        if deriv == 2:
            ysm[i] = (((y[i] - y[i - 1]) / (x[i] - x[i - 1])) - ((y[i - 1] - y[i - 2]) / (x[i - 1] - x[i - 2]))) / \
                     (x[i] - x[i - 1])
            ysm[j] = (((y[j] - y[j - 1]) / (x[j] - x[j - 1])) - ((y[j - 1] - y[j - 2]) / (x[j - 1] - x[j - 2]))) / \
                     (x[j] - x[j - 1])
        if deriv >= 3:
            raise NotImplementedError("derivatives >= 3 not implemented")

    m = 2 * n + 1  # the size of the filter window
    o = polyorder + 1  # the smoothing order
    A = np.zeros((m, o))  # A matrix
    t = np.zeros(m)
    # start smoothing
    for i in range(n, x.shape[0] - n):
        for j in range(m):
            t[j] = x[i + j - n] - x[i]
        for j in range(m):
            r = 1.0
            for k in range(o):
                A[j, k] = r
                r *= t[j]
        tA = A.transpose()  # A transposed
        tAA = np.matmul(tA, A)  # make tA.A
        tAA = np.linalg.inv(tAA)  # make (tA.A)-¹ in place
        tAAtA = np.matmul(tAA, tA)  # make (tA.A)-¹.tA

        # compute the polynomial's value at the center of the sample
        ysm[i] = 0.0
        for j in range(m):
            ysm[i] += tAAtA[deriv, j] * y[i + j - n]

    return ysm
