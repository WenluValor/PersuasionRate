import math
from bound import (ora_bound, est_bound, boots_bound, ML_bound, empirical_bound,
                   est_denum, ora_denum, ML_denum, boots_denum)
from poest import get_Yntn, setup
import numpy as np
import pandas as pd
import data as dt


def set_global(n_val, case, rho, p_val):
    global N, CASE, P, NAME
    N = n_val
    CASE = case
    P = p_val
    num = round(math.log10(N) * 10)
    NAME = 'n' + str(num) + 'r' + str(round(rho * 10))


def empirical_po(direct, natural):
    po_Y0 = np.zeros([N + 1])
    po_Y1 = np.zeros([N + 1])
    pr_Y = np.zeros([N])
    po_Yt = np.zeros([N + 1])
    if natural:
        for i in range(N + 1):
            Y0 = np.array(pd.read_csv('test-data/valid-data/'
                                      + str(i) + 'NY.csv', index_col=0))[:, 0]
            Yt = np.array(pd.read_csv('test-data/valid-data/'
                                      + str(i) + 'NY.csv', index_col=0))[:, 3]
            if direct:
                Y1 = np.array(pd.read_csv('test-data/valid-data/'
                                          + str(i) + 'NY.csv', index_col=0))[:, 1]
            else:
                Y1 = np.array(pd.read_csv('test-data/valid-data/'
                                          + str(i) + 'NY.csv', index_col=0))[:, 2]
            if i != 0:
                pr_Y[i - 1] = np.average(Y1[Y0 == 0])
            po_Y0[i] = np.average(Y0)
            po_Y1[i] = np.average(Y1)
            po_Yt[i] = np.average(Yt)
        pr = np.average(pr_Y)
    else:
        for i in range(N + 1):
            Y0 = np.array(pd.read_csv('test-data/valid-data/'
                                      + str(i) + 'ZY.csv', index_col=0))[:, 0]
            if direct:
                Y1 = np.array(pd.read_csv('test-data/valid-data/'
                                          + str(i) + 'ZY.csv', index_col=0))[:, 1]
            else:
                Y1 = np.array(pd.read_csv('test-data/valid-data/'
                                          + str(i) + 'ZY.csv', index_col=0))[:, 2]
            if i != 0:
                pr_Y[i - 1] = np.average(Y1[Y0 == 0])
            po_Y0[i] = np.average(Y0)
            po_Y1[i] = np.average(Y1)
        pr = np.average(pr_Y)
        po_Yt = po_Y0
    return po_Y0, po_Y1, pr, po_Yt


def empirical_pr(direct, natural):
    emp_Y0, emp_Y1, emp_pr, emp_Yt = empirical_po(direct=direct, natural=natural)
    true, inf, sup, RR = empirical_bound(po_Y0=emp_Y0, po_Y1=emp_Y1, pr=emp_pr)
    res = np.zeros([4])
    res[0] = true
    res[1] = inf
    res[2] = sup
    res[3] = RR
    DF = pd.DataFrame(res)
    if natural:
        if direct:
            DF.to_csv('test-data/results/' + NAME + 'dirNtrue' + '.csv')
        else:
            DF.to_csv('test-data/results/' + NAME + 'indNtrue' + '.csv')
    else:
        if direct:
            DF.to_csv('test-data/results/' + NAME + 'dirZtrue' + '.csv')
        else:
            DF.to_csv('test-data/results/' + NAME + 'indZtrue' + '.csv')
    return emp_Yt


def empirical_denum(direct, natural):
    emp_Y0, emp_Y1, emp_pr, emp_Yt = empirical_po(direct=direct, natural=natural)
    num = np.average(1 - emp_Y1)
    denum = np.average(1 - emp_Y0)
    return num, denum, emp_Yt


def pr_bound(methods: list, rep: int, direct, alpha=0.1):
    B = rep
    col_name = []
    for item in methods:
        col_name.append('inf_' + item)
        col_name.append('sup_' + item)
        col_name.append('RR_' + item)
        col_name.append('v_inf_' + item)
        col_name.append('v_sup_' + item)
        col_name.append('v_RR_' + item)

    for natural in [True, False]:
        res = pd.DataFrame(np.zeros([B, len(col_name)]), columns=col_name)
        emp_num = empirical_pr(direct=direct, natural=natural)
        for i in range(B):
            dt.gene_data(p_val=P, n_val=N, case=CASE, alpha=alpha)
            Y = np.array(pd.read_csv('test-data/Y.csv', index_col=0))[1: (N + 1)]
            T = np.array(pd.read_csv('test-data/T.csv', index_col=0))[1: (N + 1)]
            while (not ((0 in Y) & (1 in Y) & (0 in T) & (1 in T))):
                dt.gene_data(p_val=P, n_val=N, case=CASE, alpha=alpha)
                Y = np.array(pd.read_csv('test-data/Y.csv', index_col=0))[1: (N + 1)]
                T = np.array(pd.read_csv('test-data/T.csv', index_col=0))[1: (N + 1)]
                print(i)

            setup(n_val=N)
            if natural:
                T = np.insert(T, 0, 0)
                est_Y = get_Yntn(vec_t=T)
            else:
                est_Y = get_Yntn(vec_t=np.zeros([N + 1]))
            est_Y = np.where(est_Y >= 0, est_Y, 0)
            est_num = np.where(est_Y <= 1, est_Y, 1)
            F = np.array(pd.read_csv('test-data/F.csv', index_col=0))

            for method in methods:
                if method == 'est':
                    inf, sup, RR, v_inf, v_sup, v_RR = est_bound(po_num=est_num, direct=direct, natural=natural)
                elif method == 'ora':
                    inf, sup, RR, v_inf, v_sup, v_RR = ora_bound(po_num=emp_num, case=CASE, n_val=N,
                                                                 direct=direct, natural=natural)
                elif method == 'boots':
                    inf, sup, RR, v_inf, v_sup, v_RR = boots_bound(rep_B=200, sam_B=1000, po_num=est_num,
                                                                   F=F, direct=direct, natural=natural)
                elif method == 'ML':
                    inf, sup, RR, v_inf, v_sup, v_RR = ML_bound(n_val=N, direct=direct, natural=natural)
                else:
                    inf, sup, RR, v_inf, v_sup, v_RR = 0, 0, 0, 0, 0, 0

                col1 = 'inf_' + method
                res.loc[i, col1] = inf
                col2 = 'sup_' + method
                res.loc[i, col2] = sup
                col3 = 'RR_' + method
                res.loc[i, col3] = RR
                col1 = 'v_inf_' + method
                res.loc[i, col1] = v_inf
                col2 = 'v_sup_' + method
                res.loc[i, col2] = v_sup
                col3 = 'v_RR_' + method
                res.loc[i, col3] = v_RR

            if i % 10 == 0:
                print(i, B, direct)
        if natural:
            if direct:
                res.to_csv('test-data/results/' + NAME + 'dirNmethods' + '.csv')
            else:
                res.to_csv('test-data/results/' + NAME + 'indNmethods' + '.csv')
        else:
            if direct:
                res.to_csv('test-data/results/' + NAME + 'dirZmethods' + '.csv')
            else:
                res.to_csv('test-data/results/' + NAME + 'indZmethods' + '.csv')


def pr_denum(methods: list, rep: int, direct, alpha=0.1):
    B = rep
    col_name = []
    col_name.append('true_num')
    col_name.append('true_denom')
    for item in methods:
        col_name.append('num_' + item)
        col_name.append('denom_' + item)

    for natural in [True, False]:
        res = pd.DataFrame(np.zeros([B, len(col_name)]), columns=col_name)
        true_num, true_denum, emp_num = empirical_denum(direct=direct, natural=natural)
        res.loc[0, 'true_num'] = true_num
        res.loc[0, 'true_denom'] = true_denum

        for i in range(B):
            dt.gene_data(p_val=P, n_val=N, case=CASE, alpha=alpha)
            Y = np.array(pd.read_csv('test-data/Y.csv', index_col=0))[1: (N + 1)]
            T = np.array(pd.read_csv('test-data/T.csv', index_col=0))[1: (N + 1)]
            while (not ((0 in Y) & (1 in Y) & (0 in T) & (1 in T))):
                dt.gene_data(p_val=P, n_val=N, case=CASE, alpha=alpha)
                Y = np.array(pd.read_csv('test-data/Y.csv', index_col=0))[1: (N + 1)]
                T = np.array(pd.read_csv('test-data/T.csv', index_col=0))[1: (N + 1)]
                print(i)

            setup(n_val=N)
            if natural:
                T = np.insert(T, 0, 0)
                est_Y = get_Yntn(vec_t=T)
            else:
                est_Y = get_Yntn(vec_t=np.zeros([N + 1]))
            est_Y = np.where(est_Y >= 0, est_Y, 0)
            est_num = np.where(est_Y <= 1, est_Y, 1)
            F = np.array(pd.read_csv('test-data/F.csv', index_col=0))

            for method in methods:
                if method == 'est':
                    num, denom = est_denum(po_num=est_num, direct=direct, natural=natural)
                elif method == 'ora':
                    num, denom = ora_denum(po_num=emp_num, case=CASE, n_val=N,
                                           direct=direct, natural=natural)
                elif method == 'boots':
                    num, denom = boots_denum(rep_B=200, sam_B=1000, po_num=est_num,
                                             F=F, direct=direct, natural=natural)
                elif method == 'ML':
                    num, denom = ML_denum(n_val=N, direct=direct, natural=natural)
                else:
                    num, denom = 0, 0

                col1 = 'num_' + method
                res.loc[i, col1] = num
                col2 = 'denom_' + method
                res.loc[i, col2] = denom

            if i % 10 == 0:
                print(i, B, direct)
        if natural:
            if direct:
                res.to_csv('test-data/results/' + NAME + 'dirNdenum' + '.csv')
            else:
                res.to_csv('test-data/results/' + NAME + 'indNdenum' + '.csv')
        else:
            if direct:
                res.to_csv('test-data/results/' + NAME + 'dirZdenum' + '.csv')
            else:
                res.to_csv('test-data/results/' + NAME + 'indZdenum' + '.csv')


def run():
    dt.check_path()

    case = 'simu'
    rho = 0.1
    p = 500

    for i in range(0, 9):
        n_val = round(10 ** (i / 2 + 1))
        set_global(case=case, n_val=n_val, rho=rho, p_val=p)
        methods = ['est', 'ora', 'ML']
        pr_denum(methods=methods, rep=500, direct=True)
        pr_denum(methods=methods, rep=500, direct=False)

    '''
    case = 'hmm'
    rho = 0.0
    p = 10
    n = int(1e5)
    m = 1000

    dt.gene_data(p_val=p, n_val=n, case=case, alpha=0.1)
    dt.gene_valid(p_val=p, n_val=n, m_val=m, case=case, rho=rho, alpha=0.1)

    for i in range(0, 9):
        n_val = round(10 ** (i / 2 + 1))
        set_global(case=case, n_val=n_val, rho=rho, p_val=p)
        methods = ['est', 'ora', 'ML', 'boots']
        pr_bound(methods=methods, rep=500, direct=True)
        pr_bound(methods=methods, rep=500, direct=False)
        #empirical_pr(direct=True, natural=True)
        #empirical_pr(direct=True, natural=False)
        #empirical_pr(direct=False, natural=True)
        #empirical_pr(direct=False, natural=False)
    '''

    '''
    case = 'test'
    p = 5
    n = int(1e4)
    m = 100
    alpha = 0.1
    # dt.gene_valid(p_val=p, n_val=n, m_val=m, case=case, alpha=alpha, rho=0.0)

    for i in range(0, 4):
        n_val = round(10 ** (i / 2 + 1))
        set_global(case=case, n_val=n_val, rho=0.0, p_val=p)
        methods = ['est', 'ora', 'ML', 'boots']
        pr_bound(methods=methods, rep=5, alpha=alpha, direct=False)
    '''


if __name__ == '__main__':
    run()
    exit(0)
