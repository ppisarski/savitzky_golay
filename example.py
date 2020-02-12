"""
Example explaining how to use the filter
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from filter import savgol_nonuniform


def lorentzian(x, eta):
    return 1.0 / np.pi * eta / (x * x + eta * eta)


def diff_lorentzian(x, eta):
    return -2.0 / np.pi * x * eta / (x * x + eta * eta) ** 2


def gen_data(n=100):
    x = np.arange(0, n) * 0.02
    y = lorentzian(x - 1.0, 0.2) + np.random.randn(n) * 0.1
    return x, y


def main():
    x, y = gen_data()
    window_length = 7
    polyorder = 2

    ysm = savgol_nonuniform(x, y, int((window_length - 1) / 2), polyorder, 0)
    yusm = savgol_filter(y, window_length, polyorder, 0)

    fig, ax = plt.subplots(2, 1)
    fig.suptitle("Savitzky-Golay filter for window_length={} and polyorder={}".format(window_length, polyorder),
                 fontsize=14)
    ax[0].plot(x, lorentzian(x - 1.0, 0.2), color="C7", label="Lorentzian")
    ax[0].scatter(x, y, color="C0", label="data + noise")
    ax[0].plot(x, yusm, color="C1", label="Savitzky-Golay uniform")
    ax[0].plot(x, ysm, color="C2", label="Savitzky-Golay nonuniform")
    ax[0].legend(loc="best")
    ax[0].set_title("data filtering without differentiating")

    ysm = savgol_nonuniform(x, y, int((window_length - 1) / 2), polyorder, 1)
    yusm = savgol_filter(y, window_length, polyorder, 1, 0.02)

    ax[1].plot(x, diff_lorentzian(x - 1.0, 0.2), color="C7", label="derivative")
    ax[1].scatter(x, np.concatenate([[0], np.diff(y) / np.diff(x)]), color="C0", label="data + noise derivative")
    ax[1].plot(x, yusm, color="C1", label="Savitzky-Golay uniform")
    ax[1].plot(x, ysm, color="C2", label="Savitzky-Golay nonuniform")
    ax[1].legend(loc="best")
    ax[1].set_title("first derivative of filtered data")

    plt.show()


if __name__ == "__main__":
    main()
