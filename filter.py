"""
 Implements the method described in
 https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
 free to use at the user's risk
"""

__all__ = ["savgol_nonuniform"]

import numpy as np


def savgol_nonuniform(x, y, n, deg, deriv=0):
    """
    Savitzky-Golay smoothing 1D filter

    :param x:
    :param y:
    :param n: the half size of the smoothing sample, e.g. n=2 for smoothing over 5 points
    :param deg: the degree of the local polynomial fit, e.g. deg=2 for a parabolic fit
    :param deriv: The order of the derivative to compute. This must be a nonnegative integer.
            The default is 0, which means to filter the data without differentiating.
    :return:
    """
    if type(x) is not np.array:
        x = np.array(x)
    if type(y) is not np.array:
        y = np.array(y)

    if x.shape != y.shape:
        raise RuntimeError("don't even try")
    if x.shape[0] <= 2 * n:
        raise RuntimeError("not enough data to start the smoothing process")
    if 2 * n + 1 <= deg + 1:
        raise RuntimeError("need at least deg+1 points to make the polynomial")

    m = 2 * n + 1  # the size of the filter window
    o = deg + 1  # the smoothing order

    A = np.zeros((m, o))
    tA = np.zeros((o, m))

    t = np.zeros(m)
    c = np.zeros(o)

    # do not smooth start and end data
    sz = y.shape[0]
    ysm = np.zeros(y.shape)
    for i in range(n):
        ysm[i] = y[i]
        ysm[sz - i - 1] = y[sz - i - 1]

    # start smoothing
    for i in range(n, x.shape[0] - n):
        # make A and tA
        for j in range(m):
            t[j] = x[i + j - n] - x[i]
        for j in range(m):
            r = 1.0
            for k in range(o):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # make tA.A
        tAA = np.matmul(tA, A)

        # make (tA.A)-¹ in place
        tAA = np.linalg.inv(tAA)

        # make (tA.A)-¹.tA
        tAAtA = np.matmul(tAA, tA)

        # compute the polynomial's value at the center of the sample
        ysm[i] = 0.0
        for j in range(m):
            ysm[i] += tAAtA[deriv, j] * y[i + j - n]

    return ysm
