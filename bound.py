import numpy as np
import pandas as pd
import pickle
from data import f_true
from varest import get_var


def boots_bound(rep_B: int, sam_B: int, po_num, F, direct, natural):
    N = po_num.shape[0] - 1
    po_Y0 = np.zeros([N + 1])
    po_Y1 = np.zeros([N + 1])
    if natural:
        T = np.array(pd.read_csv('test-data/T.csv', index_col=0))[1: N]
    else:
        T = np.zeros([N - 1])

    if direct:
        po_Y0[1: (N + 1)] = F[0] * (1 - po_num[0: N]) + F[1] * po_num[0: N]
        po_Y1[1: (N + 1)] = F[2] * (1 - po_num[0: N]) + F[3] * po_num[0: N]
    else:
        list_F = np.zeros([4])
        F = F.reshape(-1)
        for y_val in range(2):
            list_F += (y_val * F + (1 - y_val) * (1 - F)) * F[y_val]
        T = T.astype(int)
        po_Y0[2: (N + 1)] = (list_F[2 * T].reshape(-1) * (1 - po_num[0: N - 1])
                             + list_F[2 * T + 1].reshape(-1) * po_num[0: N - 1])
        po_Y1[2: (N + 1)] = list_F[2] * (1 - po_num[0: N - 1]) + list_F[3] * po_num[0: N - 1]

    po_Y0 = np.where(po_Y0 >= 0, po_Y0, 0)
    po_Y0 = np.where(po_Y0 <= 1, po_Y0, 1)
    po_Y1 = np.where(po_Y1 >= 0, po_Y1, 0)
    po_Y1 = np.where(po_Y1 <= 1, po_Y1, 1)

    res = np.zeros([rep_B])
    v_res = np.zeros([rep_B])
    tot_Y0 = np.array([np.random.binomial(n=1, p=prob, size=sam_B) for prob in po_Y0])
    tot_Y1 = np.array([np.random.binomial(n=1, p=prob, size=sam_B) for prob in po_Y1])

    for i in range(rep_B):
        if N > sam_B:
            index0 = np.random.choice(N, size=sam_B, replace=True)
            index0 += 1
        else:
            index0 = np.arange(1, N + 1)
        index1 = np.random.choice(sam_B, size=sam_B, replace=True)

        Yb0 = tot_Y0[index0, :]
        Yb0 = Yb0[:, index1]
        Yb1 = tot_Y1[index0, :]
        Yb1 = Yb1[:, index1]

        denom = 1 - po_Y0[index0]
        num = np.average((Yb0 == 0) & (Yb1 == 1), axis=1)
        tmp = np.where(denom != 0, denom, 1)
        ans = np.where(denom != 0, num / tmp, 0)

        res[i] = np.average(ans)
        v_res[i] = np.var(ans)

        #denom = 1 - po_Y0[index0]
        #num = np.average((Yb1 == 0), axis=1)
        #tmp = np.where(denom != 0, denom, 1)
        #ans = np.where(denom != 0, num / tmp, 0)
        #RR_res[i] = np.average(ans)
        #v_RR_res[i] = np.var(ans)


    inf = np.min(res)
    sup = np.max(res)
    v_inf = np.average(v_res)
    v_sup = np.average(v_res)

    #RR = np.average(RR_res)
    #v_RR = np.average(v_RR_res)
    RR, v_RR = 0, 0
    return max(inf, 0), min(1, sup), RR, v_inf, v_sup, v_RR


def MC_hoeff_bound(po_Y0, po_Y1):
    # pr = P(Y1 = 1 | Y0 = 0)
    prob_y0 = 1 - po_Y0
    prob_y1 = po_Y1
    tmp = np.where(prob_y1 + prob_y0 - 1 > 0, prob_y1 + prob_y0 - 1, 0)
    inf = np.average(tmp / prob_y0)
    # v_inf = np.var(tmp / prob_y0)

    tmp = np.where(prob_y1 > prob_y0, prob_y0, prob_y1)
    sup = np.average(tmp / prob_y0)
    # v_sup = np.var(tmp / prob_y0)

    return inf, sup


def theta_bound(cov_F, po_num, F, direct, natural):
    '''
    :param po_num: numerator P(Y(0/t)=1) vector through 0 to n
    :param F:
    :param direct:
    :return:
    '''
    N = po_num.shape[0] - 1
    L_theta = np.zeros([2])
    U_theta = np.zeros([2])

    if direct:
        for i in range(2):
            L_theta[i] = max(F[i + 2] - F[i], 0)
            U_theta[i] = min(F[i + 2], 1 - F[i])
        num1 = po_num[0: N] * L_theta[1] + (1 - po_num[0: N]) * L_theta[0]
        num2 = po_num[0: N] * U_theta[1] + (1 - po_num[0: N]) * U_theta[0]
        num3 = 1 - (po_num[0: N] * F[3] + (1 - po_num[0: N]) * F[2])
        denom = 1 - (po_num[0: N] * F[1] + (1 - po_num[0: N]) * F[0])
    else:
        list_F = np.zeros([4])
        F = F.reshape(-1)
        # only consider tilde_F(0, t; y)
        for y_val in range(2):
            list_F += (y_val * F + (1 - y_val) * (1 - F)) * F[y_val]

        for i in range(2):
            L_theta[i] = max(list_F[i + 2] - list_F[i], 0)
            U_theta[i] = min(list_F[i + 2], 1 - list_F[i])
        num1 = po_num[0: N - 1] * L_theta[1] + (1 - po_num[0: N - 1]) * L_theta[0]
        num2 = po_num[0: N - 1] * U_theta[1] + (1 - po_num[0: N - 1]) * U_theta[0]
        num3 = 1 - (po_num[0: N - 1] * list_F[3] + (1 - po_num[0: N - 1]) * list_F[2])
        if natural:
            T = np.array(pd.read_csv('test-data/T.csv', index_col=0))[1: N]
        else:
            T = np.zeros([N - 1])
        T = T.astype(int)
        denom = 1 - (po_num[0: N - 1] * list_F[2 * T + 1].reshape(-1)
                     + (1 - po_num[0: N - 1]) * list_F[2 * T].reshape(-1))

    tmp = np.where(denom != 0, denom, 1)
    ans1 = np.where(denom != 0, num1 / tmp, 0)
    ans2 = np.where(denom != 0, num2 / tmp, 0)
    ans3 = np.where(denom != 0, num3 / tmp, 0)

    inf = np.average(ans1)
    sup = np.average(ans2)
    RR = np.average(ans3)

    v_inf = get_var(cov_F=cov_F, type='inf', po_num=po_num, F=F, direct=direct, natural=natural)
    v_sup = get_var(cov_F=cov_F, type='sup', po_num=po_num, F=F, direct=direct, natural=natural)
    v_RR = get_var(cov_F=cov_F, type='RR', po_num=po_num, F=F, direct=direct, natural=natural)

    return max(0, inf), min(1, sup), RR, v_inf, v_sup, v_RR



def ML_bound(n_val, direct, natural):
    N = n_val
    X = np.array(pd.read_csv('test-data/X.csv', index_col=0))[0: (N + 1)]

    pkl_filename = 'test-data/f_model.pkl'
    with open(pkl_filename, 'rb') as file:
        f_model = pickle.load(file)
    if f_model.classes_[0] == 0:
        f_ordered = True
    else:
        f_ordered = False

    values = [[0, 0], [0, 1], [1, 0], [1, 1]]
    hat_F = np.zeros([4])
    tmp_F = np.zeros([N + 1, 4])
    for i in range(4):
        data = np.concatenate((np.zeros([N + 1, 1]) + values[i][0],
                               np.zeros([N + 1, 1]) + values[i][1], X), axis=1)
        hat_F[i] = np.average(f_model.predict_proba(data)[:, int(f_ordered)])
        tmp_F[:, i] = f_model.predict_proba(data)[:, int(f_ordered)]
    hat_F = np.where(hat_F < 0, 0, hat_F)
    hat_F = np.where(hat_F > 1, 1, hat_F)

    po_num = np.zeros([N + 1])
    if natural:
        T = np.array(pd.read_csv('test-data/T.csv', index_col=0))[1: (N + 1)]
    else:
        T = np.zeros([N])

    for i in range(1, N + 1):
        t = int(T[i - 1])
        po_num[i] = (hat_F[2 * t] * (1 - po_num[i - 1]) + hat_F[2 * t + 1] * po_num[i - 1])

    cov_F = np.cov(tmp_F, rowvar=False, bias=True)
    inf, sup, RR, v_inf, v_sup, v_RR \
        = theta_bound(cov_F=cov_F, po_num=po_num, direct=direct, F=hat_F, natural=natural)
    return inf, sup, RR, v_inf, v_sup, v_RR


def est_bound(po_num, direct, natural):
    F = np.array(pd.read_csv('test-data/F.csv', index_col=0))
    cov_F = np.array(pd.read_csv('test-data/var_F.csv', index_col=0))
    inf, sup, RR, v_inf, v_sup, v_RR \
        = theta_bound(cov_F=cov_F, po_num=po_num, direct=direct, F=F, natural=natural)
    return inf, sup, RR, v_inf, v_sup, v_RR


def ora_bound(po_num, case, n_val, direct, natural):
    CASE = case
    N = n_val
    X = np.array(pd.read_csv('test-data/X.csv', index_col=0))[0: (N + 1)]
    F = np.zeros([4])
    tmp_F = np.zeros([N + 1, 4])

    for y_val in range(2):
        if CASE == 'hmm':
            U = np.array(pd.read_csv('test-data/U.csv', index_col=0))[0: (N + 1)]
            F[y_val] = np.average(f_true(x=X, case=CASE, t=0, y=y_val, u=U), axis=0)
            F[y_val + 2] = np.average(f_true(x=X, case=CASE, t=1, y=y_val, u=U), axis=0)
            tmp_F[:, y_val] = f_true(x=X, case=CASE, t=0, y=y_val, u=U)
            tmp_F[:, y_val + 2] = f_true(x=X, case=CASE, t=1, y=y_val, u=U)
        else:
            F[y_val] = np.average(f_true(x=X, case=CASE, t=0, y=y_val), axis=0)
            F[y_val + 2] = np.average(f_true(x=X, case=CASE, t=1, y=y_val), axis=0)
            tmp_F[:, y_val] = f_true(x=X, case=CASE, t=0, y=y_val)
            tmp_F[:, y_val + 2] = f_true(x=X, case=CASE, t=1, y=y_val)

    F = np.where(F < 0, 0, F)
    F = np.where(F > 1, 1, F)
    cov_F = np.cov(tmp_F, rowvar=False, bias=True)
    inf, sup, RR, v_inf, v_sup, v_RR \
        = theta_bound(cov_F=cov_F, po_num=po_num, direct=direct, F=F, natural=natural)
    return inf, sup, RR, v_inf, v_sup, v_RR


def empirical_bound(po_Y0, po_Y1, pr):
    # pr = P(Y1 = 1 | Y0 = 0)
    N = po_Y0.shape[0] - 1
    true = pr
    inf, sup = MC_hoeff_bound(po_Y0=po_Y0[1: N + 1], po_Y1=po_Y1[1: N + 1])
    RR = np.average((1 - po_Y1[1: N + 1]) / (1 - po_Y0[1: N + 1]))
    # v_RR = np.var((1 - po_Y1[1: N + 1]) / (1 - po_Y0[1: N + 1]))
    return true, inf, sup, RR


def theta_denum(po_num, F, direct, natural):
    N = po_num.shape[0] - 1
    if direct:
        num = 1 - (po_num[0: N] * F[3] + (1 - po_num[0: N]) * F[2])
        denom = 1 - (po_num[0: N] * F[1] + (1 - po_num[0: N]) * F[0])
    else:
        list_F = np.zeros([4])
        F = F.reshape(-1)
        # only consider tilde_F(0, t; y)
        for y_val in range(2):
            list_F += (y_val * F + (1 - y_val) * (1 - F)) * F[y_val]

        num = 1 - (po_num[0: N - 1] * list_F[3] + (1 - po_num[0: N - 1]) * list_F[2])
        if natural:
            T = np.array(pd.read_csv('test-data/T.csv', index_col=0))[1: N]
        else:
            T = np.zeros([N - 1])
        T = T.astype(int)
        denom = 1 - (po_num[0: N - 1] * list_F[2 * T + 1].reshape(-1)
                     + (1 - po_num[0: N - 1]) * list_F[2 * T].reshape(-1))
    ans1 = np.average(num)
    ans2 = np.average(denom)
    return ans1, ans2


def ML_denum(n_val, direct, natural):
    N = n_val
    X = np.array(pd.read_csv('test-data/X.csv', index_col=0))[0: (N + 1)]

    pkl_filename = 'test-data/f_model.pkl'
    with open(pkl_filename, 'rb') as file:
        f_model = pickle.load(file)
    if f_model.classes_[0] == 0:
        f_ordered = True
    else:
        f_ordered = False

    values = [[0, 0], [0, 1], [1, 0], [1, 1]]
    hat_F = np.zeros([4])
    for i in range(4):
        data = np.concatenate((np.zeros([N + 1, 1]) + values[i][0],
                               np.zeros([N + 1, 1]) + values[i][1], X), axis=1)
        hat_F[i] = np.average(f_model.predict_proba(data)[:, int(f_ordered)])
    hat_F = np.where(hat_F < 0, 0, hat_F)
    hat_F = np.where(hat_F > 1, 1, hat_F)

    po_num = np.zeros([N + 1])
    if natural:
        T = np.array(pd.read_csv('test-data/T.csv', index_col=0))[1: (N + 1)]
    else:
        T = np.zeros([N])

    for i in range(1, N + 1):
        t = int(T[i - 1])
        po_num[i] = (hat_F[2 * t] * (1 - po_num[i - 1]) + hat_F[2 * t + 1] * po_num[i - 1])

    num, denom = theta_denum(po_num=po_num, direct=direct, F=hat_F, natural=natural)
    return num, denom


def est_denum(po_num, direct, natural):
    F = np.array(pd.read_csv('test-data/F.csv', index_col=0))
    num, denom = theta_denum(po_num=po_num, direct=direct, F=F, natural=natural)
    return num, denom


def ora_denum(po_num, case, n_val, direct, natural):
    CASE = case
    N = n_val
    X = np.array(pd.read_csv('test-data/X.csv', index_col=0))[0: (N + 1)]
    F = np.zeros([4])
    tmp_F = np.zeros([N + 1, 4])

    for y_val in range(2):
        if CASE == 'hmm':
            U = np.array(pd.read_csv('test-data/U.csv', index_col=0))[0: (N + 1)]
            F[y_val] = np.average(f_true(x=X, case=CASE, t=0, y=y_val, u=U), axis=0)
            F[y_val + 2] = np.average(f_true(x=X, case=CASE, t=1, y=y_val, u=U), axis=0)
            tmp_F[:, y_val] = f_true(x=X, case=CASE, t=0, y=y_val, u=U)
            tmp_F[:, y_val + 2] = f_true(x=X, case=CASE, t=1, y=y_val, u=U)
        else:
            F[y_val] = np.average(f_true(x=X, case=CASE, t=0, y=y_val), axis=0)
            F[y_val + 2] = np.average(f_true(x=X, case=CASE, t=1, y=y_val), axis=0)
            tmp_F[:, y_val] = f_true(x=X, case=CASE, t=0, y=y_val)
            tmp_F[:, y_val + 2] = f_true(x=X, case=CASE, t=1, y=y_val)

    F = np.where(F < 0, 0, F)
    F = np.where(F > 1, 1, F)

    num, denom = theta_denum(po_num=po_num, direct=direct, F=F, natural=natural)
    return num, denom


def boots_denum(rep_B: int, sam_B: int, po_num, F, direct, natural):
    N = po_num.shape[0] - 1
    po_Y0 = np.zeros([N + 1])
    po_Y1 = np.zeros([N + 1])
    if natural:
        T = np.array(pd.read_csv('test-data/T.csv', index_col=0))[1: N]
    else:
        T = np.zeros([N - 1])

    if direct:
        po_Y0[1: (N + 1)] = F[0] * (1 - po_num[0: N]) + F[1] * po_num[0: N]
        po_Y1[1: (N + 1)] = F[2] * (1 - po_num[0: N]) + F[3] * po_num[0: N]
    else:
        list_F = np.zeros([4])
        F = F.reshape(-1)
        for y_val in range(2):
            list_F += (y_val * F + (1 - y_val) * (1 - F)) * F[y_val]
        T = T.astype(int)
        po_Y0[2: (N + 1)] = (list_F[2 * T].reshape(-1) * (1 - po_num[0: N - 1])
                             + list_F[2 * T + 1].reshape(-1) * po_num[0: N - 1])
        po_Y1[2: (N + 1)] = list_F[2] * (1 - po_num[0: N - 1]) + list_F[3] * po_num[0: N - 1]

    po_Y0 = np.where(po_Y0 >= 0, po_Y0, 0)
    po_Y0 = np.where(po_Y0 <= 1, po_Y0, 1)
    po_Y1 = np.where(po_Y1 >= 0, po_Y1, 0)
    po_Y1 = np.where(po_Y1 <= 1, po_Y1, 1)

    num_res = np.zeros([rep_B])
    denom_res = np.zeros([rep_B])

    tot_Y1 = np.array([np.random.binomial(n=1, p=prob, size=sam_B) for prob in po_Y1])

    for i in range(rep_B):
        if N > sam_B:
            index0 = np.random.choice(N, size=sam_B, replace=True)
            index0 += 1
        else:
            index0 = np.arange(1, N + 1)
        index1 = np.random.choice(sam_B, size=sam_B, replace=True)

        Yb1 = tot_Y1[index0, :]
        Yb1 = Yb1[:, index1]

        denom = 1 - po_Y0[index0]
        num = np.average((Yb1 == 0), axis=1)
        denom_res[i] = np.average(denom)
        num_res[i] = np.average(num)

    ans1 = np.average(num_res)
    ans2 = np.average(denom_res)

    return ans1, ans2

