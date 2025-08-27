import numpy as np
import pandas as pd

def est_Di(po_num, F, natural):
    # return derivation P(Y(t) = 1)
    N = po_num.shape[0] - 1
    if natural:
        T = np.array(pd.read_csv('real-data/T.csv', index_col=0))[1: (N + 1)]
        T = np.insert(T, 0, 0)
    else:
        T = np.zeros([N + 1])


    T = T.astype(int)
    res = np.zeros([N + 1, 4])
    for i in range(1, N + 1):
        if i == 1:
            res[i] = ((2 * T[i]) == np.arange(4)).astype(int) * (1 - po_num[i - 1])
        else:
            res[i] = ((2 * T[i]) == np.arange(4)).astype(int) * (1 - po_num[i - 1]) \
                     + ((2 * T[i] + 1) == np.arange(4)).astype(int) * (po_num[i - 1]) \
                     - F[2 * T[i]] * res[i - 1] + F[2 * T[i] + 1] * res[i - 1]
    return res


def est_de_num(po_num, F, natural, direct):
    # P(Y1 = 1| Y0 = 1), return derivative D0 = P(Y0 = 1), D1 = P(Y0 = 1)
    N = po_num.shape[0] - 1
    D = est_Di(po_num, F, natural)
    D0 = np.zeros([N + 1, 4])
    D1 = np.zeros([N + 1, 4])

    for i in range(1, N + 1):
        if i == 1:
            D0[i] = (0 == np.arange(4)).astype(int) * (1 - po_num[i - 1])
            D1[i] = (2 == np.arange(4)).astype(int) * (1 - po_num[i - 1])
        else:
            D0[i] = (0 == np.arange(4)).astype(int) * (1 - po_num[i - 1]) \
                    + (1 == np.arange(4)).astype(int) * (po_num[i - 1]) \
                    - F[0] * D[i - 1] + F[1] * D[i - 1]
            D1[i] = (2 == np.arange(4)).astype(int) * (1 - po_num[i - 1]) \
                    + (3 == np.arange(4)).astype(int) * (po_num[i - 1]) \
                    - F[2] * D[i - 1] + F[3] * D[i - 1]
    if direct:
        return D1, D0
    else:
        D2 = np.zeros([N + 1, 4])
        po_num = po_num[0: N] * F[3] + (1 - po_num[0: N]) * F[2]
        po_num = np.insert(po_num, 0, 0)
        for i in range(1, N + 1):
            if i == 1:
                D2[i] = (0 == np.arange(4)).astype(int) * (1 - po_num[i - 1])
            else:
                D2[i] = (0 == np.arange(4)).astype(int) * (1 - po_num[i - 1]) \
                        + (1 == np.arange(4)).astype(int) * (po_num[i - 1]) \
                        - F[0] * D1[i - 1] + F[1] * D1[i - 1]
        return D2, D0


def get_deriv(type, po_num, F, natural, direct):
    N = po_num.shape[0] - 1
    # D1 is numerator; D0 is denominator
    D1, D0 = est_de_num(po_num, F, natural, direct)
    D1 = np.delete(D1, 0, 0)
    D0 = np.delete(D0, 0, 0)

    if direct:
        LU_theta = np.zeros([2])
        for i in range(2):
            if type == 'inf':
                LU_theta[i] = max(F[i + 2] - F[i], 0)
            elif type == 'sup':
                LU_theta[i] = min(F[i + 2], 1 - F[i])
        if type == 'RR':
            num = 1 - (po_num[0: N] * F[3] + (1 - po_num[0: N]) * F[2])
        else:
            num = po_num[0: N] * LU_theta[1] + (1 - po_num[0: N]) * LU_theta[0]
        denom = 1 - (po_num[0: N] * F[1] + (1 - po_num[0: N]) * F[0])
    else:
        LU_theta = np.zeros([2])
        list_F = np.zeros([4])
        F = F.reshape(-1)
        # only consider tilde_F(0, t; y)
        for y_val in range(2):
            list_F += (y_val * F + (1 - y_val) * (1 - F)) * F[y_val]

        for i in range(2):
            if type == 'inf':
                LU_theta[i] = max(list_F[i + 2] - list_F[i], 0)
            elif type == 'sup':
                LU_theta[i] = min(list_F[i + 2], 1 - list_F[i])
        if type == 'RR':
            num = 1 - (po_num[0: N - 1] * list_F[3] + (1 - po_num[0: N - 1]) * list_F[2])
            num = np.insert(num, 0, 1)
        else:
            num = po_num[0: N - 1] * LU_theta[1] + (1 - po_num[0: N - 1]) * LU_theta[0]
            num = np.insert(num, 0, 0)
        if natural:
            T = np.array(pd.read_csv('real-data/T.csv', index_col=0))[1: N]
        else:
            T = np.zeros([N - 1])
        T = T.astype(int)
        denom = 1 - (po_num[0: N - 1] * list_F[2 * T + 1].reshape(-1)
                     + (1 - po_num[0: N - 1]) * list_F[2 * T].reshape(-1))
        denom = np.insert(denom, 0, 1)

    tmp = np.zeros([N, 4])
    tmp_denom = np.where(denom != 0, denom, 1)
    if type == 'inf':
        for j in range(4):
            tmp[:, j] = np.where(denom != 0, 1 / tmp_denom * (D1[:, j] - D0[:, j])
                                 + num / np.square(tmp_denom) * D0[:, j], 0)
    elif type == 'sup':
        for j in range(4):
            tmp[:, j] = np.where(denom != 0, 1 / tmp_denom * np.where(D1[:, j] < - D0[:, j], D1[:, j], - D0[:, j])
                                 + num / np.square(tmp_denom) * D0[:, j], 0)
    else:
        for j in range(4):
            tmp[:, j] = np.where(denom != 0, -1 / tmp_denom * D1[:, j]
                                 + num / np.square(tmp_denom) * D0[:, j], 0)
    res = np.average(tmp, axis=0)
    return res


def get_var(cov_F, type, po_num, F, natural, direct):
    N = po_num.shape[0] - 1
    grad_h = get_deriv(type, po_num, F, natural, direct)
    var_F = np.dot(np.dot(grad_h, cov_F), grad_h.T) / N
    return var_F
