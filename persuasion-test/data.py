import math
import numpy as np
import pandas as pd
import os


def get_X(p_val: int, n_val: int):
    N = n_val
    P = p_val
    X = np.random.normal(loc=0, scale=1, size=(N + 1, P))

    DF = pd.DataFrame(X)
    DF.to_csv('test-data/X.csv')


def get_YT(n_val: int):
    N = n_val
    X = np.array(pd.read_csv('test-data/X.csv', index_col=0))
    T = np.zeros([N + 1])
    Y = np.zeros([N + 1])

    for i in range(1, N + 1):
        prob_T = pi_true(x=X[i], t=T[i - 1], y=Y[i - 1])
        T[i] = np.random.binomial(n=1, p=prob_T, size=1)[0]
        prob_Y = f_true(x=X[i], t=T[i], y=Y[i - 1])
        Y[i] = np.random.binomial(n=1, p=prob_Y, size=1)[0]

    DF = pd.DataFrame(T)
    DF.to_csv('test-data/T.csv')
    DF = pd.DataFrame(Y)
    DF.to_csv('test-data/Y.csv')


def pi_true(x, t, y):
    a = 0.5 + 0.1 * (2 * t - 1 + 2 * y - 1)

    if len(x.shape) == 1:
        P = x.shape[0]
        z = a + sum(x) * 0.1 / P
        z = min(1, z)
        z = max(0, z)
        return z
    else:
        P = x.shape[1]
        z = a + np.sum(x, axis=1) * 0.1 / P
        z = np.where(z > 0, z, 0)
        z = np.where(z < 1, z, 1)
        return z


def f_true(x, t, y):
    a = 0.75 - 0.1 * t - 0.2 * y + 0.3 * t * y
    if len(x.shape) == 1:
        P = x.shape[0]
        z = a + sum(x) * 0.1 / P
        z = min(1, z)
        z = max(0, z)
        return z
    else:
        P = x.shape[1]
        z = a + np.sum(x, axis=1) * 0.1 / P
        z = np.where(z > 0, z, 0)
        z = np.where(z < 1, z, 1)
        return z


def gene_data(p_val: int, n_val: int):
    get_X(p_val=p_val, n_val=n_val)
    get_YT(n_val=n_val)
    return


def check_path():
    paths = ['test-data', 'test-data/results']
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


if __name__ == '__main__':
    check_path()
    p = 5
    n = int(1e4)
    m = 500
    gene_data(p_val=p, n_val=n)
    exit(0)


