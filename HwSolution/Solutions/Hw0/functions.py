# !/usr/local/bin/python

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import statistics


def func(x_):
    y = x_ + np.sqrt(x_) + 1/np.power(x_, 2) + 10*np.sin(x_)
    return y
