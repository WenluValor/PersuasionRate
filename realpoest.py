import numpy as np
import pandas as pd
import pickle



def set_global(n_val):
    global CASE, N, X, T, Y
    N = n_val
    X = np.array(pd.read_csv('real-data/X.csv', index_col=0))[0: (N + 1)]
    T = np.array(pd.read_csv('real-data/T.csv', index_col=0))[0: (N + 1)]
    Y = np.array(pd.read_csv('real-data/Y.csv', index_col=0))[0: (N + 1)]


def get_fpi_res():
    # f(t=0, y=0), f(t=0, y=1), f(t=1, y=0), f(t=1, y=1)
    f_filename = 'real-data/f_model.pkl'
    with open(f_filename, 'rb') as file:
        f_model = pickle.load(file)
    pi_filename = 'real-data/pi_model.pkl'
    with open(pi_filename, 'rb') as file:
        pi_model = pickle.load(file)

    f_res = np.zeros([N + 1, 4])
    pi_res = np.zeros([N + 1, 4])

    if f_model.classes_[0] == 0:
        f_ordered = True
    else:
        f_ordered = False

    if pi_model.classes_[0] == 0:
        pi_ordered = True
    else:
        pi_ordered = False

    values = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for i in range(4):
        data = np.concatenate((np.zeros([N + 1, 1]) + values[i][0],
                               np.zeros([N + 1, 1]) + values[i][1], X), axis=1)
        f_res[:, i] = f_model.predict_proba(data)[:, int(f_ordered)]

        pi_res[:, i] = pi_model.predict_proba(data)[:, int(pi_ordered)]

    return f_res, pi_res


def get_A_res():
    res = np.zeros([4])  # g

    T_prime = T[1: N + 1]
    Y_prime = Y[1: N + 1]

    res[0] = len(np.where((T_prime == 0) & (Y_prime == 0))[0])
    res[1] = len(np.where((T_prime == 0) & (Y_prime == 1))[0])
    res[2] = len(np.where((T_prime == 1) & (Y_prime == 0))[0])
    res[3] = len(np.where((T_prime == 1) & (Y_prime == 1))[0])

    return res / N


def update_G():
    G = np.zeros([N + 1, 4])
    f_res, pi_res = get_fpi_res()
    A_res = get_A_res()
    ind_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
    # G1(t=0, y=0), G1(t=0, y=1), G1(t=1, y=0), G1(t=1, y=1)

    for i in range(1, N + 1):
        for j in range(len(ind_list)):
            t = int(ind_list[j][0])
            y = int(ind_list[j][1])

            for t0 in range(2):
                pos = int(ind_list.index((t0, y)))
                G[i, j] += (t * pi_res[i, pos] + (1 - t) * (1 - pi_res[i, pos])) \
                            * (A_res[pos])
    #DF = pd.DataFrame(G)
    #DF.to_csv('real-data/G.csv')
    #DF = pd.DataFrame(pi_res)
    #DF.to_csv('real-data/pi_res.csv')
    return f_res, G


def update_F():
    F = np.zeros([N + 1, 4])
    f_res, G = update_G()
    tmp_F = np.zeros([N + 1, 4])

    T0_pos = np.where(T == 0)[0]
    T1_pos = np.where(T == 1)[0]
    Y0_pos = np.where(Y == 0)[0]
    Y1_pos = np.where(Y == 1)[0]
    pos_list = [[T0_pos, T1_pos], [Y0_pos, Y1_pos]]
    ind_list = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for j in range(len(ind_list)):
        t = int(ind_list[j][0])
        y = int(ind_list[j][1])

        set_A = np.intersect1d(pos_list[0][t], pos_list[1][y] + 1)
        set_B = np.intersect1d(set_A, Y1_pos)

        ind_A = np.zeros([N + 1])
        ind_A[list(set_A)] = 1
        ind_B = np.zeros([N + 1])
        ind_B[list(set_B)] = 1

        tmp = np.where(G[:, j] != 0, G[:, j], 1)
        tmp_F[:, j] = np.where(G[:, j] != 0,
                               ind_B / tmp + (G[:, j] - ind_A) / tmp * f_res[:, j], 0)
        F_no_zeros = np.where(tmp_F[:, j] == 0, np.nan, tmp_F[:, j])


        # Calculate the cumulative sum while ignoring NaN (formerly 0 values)
        tmp = np.arange(N + 1)
        tmp[0] = 1

        F[:, j] = np.nancumsum(F_no_zeros) / tmp
        F[:, j] = np.where(F[:, j] < 0, 0, F[:, j])
        F[:, j] = np.where(F[:, j] > 1, 1, F[:, j])

    DF = pd.DataFrame(F[N])
    DF.to_csv('real-data/F.csv')

    var_F = np.cov(tmp_F, rowvar=False, bias=True)
    DF = pd.DataFrame(var_F)
    DF.to_csv('real-data/var_F.csv')
    return


def get_Yntn(vec_t: np.array):
    F = np.array(pd.read_csv('real-data/F.csv', index_col=0))
    res_Y = np.zeros([N + 1])
    # res_Y[0] = F[int(2 * vec_t[0])]

    for i in range(1, N + 1):
        t = int(vec_t[i])
        ind = 2 * t
        res_Y[i] = F[ind] * (1 - res_Y[i - 1]) + F[ind + 1] * res_Y[i - 1]

    return res_Y


def setup(n_val):
    set_global(n_val=n_val)
    update_F()
