"""
Example explaining how to use the filter
"""

import numpy as np
import matplotlib.pyplot as plt

from filter import *


def lorentzian(x, eta):
    return 1.0 / np.pi * eta / (x * x + eta * eta)


def diff_lorentzian(x, eta):
    return -2.0 / np.pi * x * eta / (x * x + eta * eta)


def gen_data(n=100):
    x = np.arange(0, n) * 0.02
    y = lorentzian(x - 1.0, 0.2) + np.random.randn(n) * 0.1
    return x, y


def main():
    x, y = gen_data()
    ysm = savgol_nonuniform(x, y, 2, 2, 1)

    plt.figure()
    plt.plot(x, diff_lorentzian(x - 1.0, 0.2), color="C1", label="Original function")
    plt.scatter(x, np.concatenate([[0], np.diff(y)]), color="C0", label="Generated data with noise")
    plt.plot(x, ysm, color="C0", label="Smoothed data")
    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    main()
