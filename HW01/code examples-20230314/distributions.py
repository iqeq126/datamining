import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels import robust
import scipy.stats as stats
import random
import operator as op
from functools import reduce


def run_distbutions():
    #Uniform
    x = np.arange(-100, 100)  # define range of x
    for ls in [(-50, 50), (10, 20)]:
        a, b = ls[0], ls[1]
        x, y, u, s = uniform(x, a, b)
        plt.plot(x, y, label=r'$\mu=%.2f,\ \sigma=%.2f$' % (u, s))

    plt.legend()
    plt.show()


    # bernoulli

    n_experiment = 100
    p = 0.6
    x = np.arange(n_experiment)
    y = []
    for _ in range(n_experiment):
        pick = bernoulli(p, k=bool(random.getrandbits(1)))
        y.append(pick)

    u, s = np.mean(y), np.std(y)
    plt.scatter(x, y, label=r'$\mu=%.2f,\ \sigma=%.2f$' % (u, s))
    plt.legend()
    plt.show()

    # normal

    x = np.arange(-100, 100)  # define range of x
    x, y, u, s = normal(x, 10000)

    plt.plot(x, y, label=r'$\mu=%.2f,\ \sigma=%.2f$' % (u, s))
    plt.legend()
    plt.show()

    # Binomial
    for ls in [(0.5, 20), (0.7, 40), (0.5, 40)]:
        p, n_experiment = ls[0], ls[1]
        x = np.arange(n_experiment)
        y, u, s = binomial(n_experiment, p)
        plt.scatter(x, y, label=r'$\mu=%.2f,\ \sigma=%.2f$' % (u, s))

    plt.legend()
    plt.show()

    #exponential
    for lamb in [0.5, 1, 1.5]:
        x = np.arange(0, 20, 0.01, dtype=np.float)
        x, y, u, s = exponential(x, lamb=lamb)
        plt.plot(x, y, label=r'$\mu=%.2f,\ \sigma=%.2f,'
                             r'\ \lambda=%d$' % (u, s, lamb))
    plt.legend()
    plt.show()

def uniform(x, a, b):

    y = [1 / (b - a) if a <= val and val <= b
                    else 0 for val in x]

    return x, y, np.mean(y), np.std(y)

def bernoulli(p, k):
    return p if k else 1 - p

def normal(x, n):
    u = x.mean()
    s = x.std()

    # normalization
    x = (x - u) / s

    # divide [x.min(), x.max()] by n
    x = np.linspace(x.min(), x.max(), n)

    a = ((x - 0) ** 2) / (2 * (1 ** 2))
    y = 1 / (s * np.sqrt(2 * np.pi)) * np.exp(-a)

    return x, y, x.mean(), x.std()

def const(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

def binomial(n, p):
    q = 1 - p
    y = [const(n, k) * (p ** k) * (q ** (n-k)) for k in range(n)]
    return y, np.mean(y), np.std(y)

def exponential(x, lamb):
    y = lamb * np.exp(-lamb * x)
    return x, y, np.mean(y), np.std(y)