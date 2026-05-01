import math
from bound import (ora_bound, est_bound, boots_bound, ML_bound, empirical_bound)
from poest import get_Yntn, setup
import numpy as np
import pandas as pd
import data as dt


def set_global(n_val, p_val):
    global N, P, NAME
    N = n_val
    P = p_val
    num = round(math.log10(N) * 10)
    NAME = 'n' + str(num)


def pr_bound(methods: list, rep: int, direct):
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
        for i in range(B):
            dt.gene_data(p_val=P, n_val=N)
            Y = np.array(pd.read_csv('test-data/Y.csv', index_col=0))[1: (N + 1)]
            T = np.array(pd.read_csv('test-data/T.csv', index_col=0))[1: (N + 1)]
            while (not ((0 in Y) & (1 in Y) & (0 in T) & (1 in T))):
                dt.gene_data(p_val=P, n_val=N)
                Y = np.array(pd.read_csv('test-data/Y.csv', index_col=0))[1: (N + 1)]
                T = np.array(pd.read_csv('test-data/T.csv', index_col=0))[1: (N + 1)]
                print(i)

            setup(n_val=N)
            F = np.array(pd.read_csv('test-data/F.csv', index_col=0))
            if natural:
                T = np.insert(T, 0, 0)
                est_Y = get_Yntn(vec_t=T, F=F)
            else:
                est_Y = get_Yntn(vec_t=np.zeros([N + 1]), F=F)
            est_Y = np.where(est_Y >= 0, est_Y, 0)
            est_num = np.where(est_Y <= 1, est_Y, 1)

            for method in methods:
                if method == 'est':
                    inf, sup, RR, v_inf, v_sup, v_RR = est_bound(po_num=est_num, direct=direct, natural=natural)
                elif method == 'ora':
                    inf, sup, RR, v_inf, v_sup, v_RR = ora_bound(n_val=N, direct=direct, natural=natural)
                elif method == 'boots':
                    inf, sup, RR, v_inf, v_sup, v_RR = boots_bound(rep_B=200, sam_B=1000, po_num=est_num,
                                                                   F=F, direct=direct, natural=natural)
                elif method == 'ML':
                    inf, sup, RR, v_inf, v_sup, v_RR = ML_bound(n_val=N, direct=direct, natural=natural)
                elif method == 'true':
                    inf, sup, RR, v_inf, v_sup, v_RR = empirical_bound(n_val=N, direct=direct, natural=natural)
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


def run():
    dt.check_path()
    p = 5

    for i in range(0, 7):
        n_val = round(10 ** (i / 2 + 1))
        set_global(n_val=n_val, p_val=p)
        # methods = ['est', 'ora', 'ML', 'boots']
        methods = ['true', 'est', 'ora', 'ML', 'boots']
        # methods = ['ora']
        pr_bound(methods=methods, rep=500, direct=True)
        pr_bound(methods=methods, rep=500, direct=False)



if __name__ == '__main__':
    run()
    exit(0)
