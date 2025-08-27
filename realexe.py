
import os

from realbound import est_bound, boots_bound, ML_bound
from realpoest import get_Yntn, setup
import numpy as np
import pandas as pd
import data as dt
from realdata import balance_T, making_T, concat_by_time

def set_global(n_val, ida, idb):
    global N, NAME
    N = n_val
    NAME = 'N' + str(N) + 'alg' + str(ida) + '-' + str(idb)


def pr_bound(methods: list, direct):
    B = 1
    col_name = []
    for item in methods:
        col_name.append('inf_' + item)
        col_name.append('sup_' + item)
        col_name.append('RR_' + item)
        col_name.append('v_inf_' + item)
        col_name.append('v_sup_' + item)
        col_name.append('v_RR_' + item)

    for natural in True, False:
        res = pd.DataFrame(np.zeros([B, len(col_name)]), columns=col_name)
        for i in range(B):
            setup(n_val=N)
            if natural:
                T = np.array(pd.read_csv('real-data/T.csv', index_col=0))[0: (N + 1)]
                est_Y = get_Yntn(vec_t=T)
            else:
                est_Y = get_Yntn(vec_t=np.zeros([N + 1]))
            est_Y = np.where(est_Y >= 0, est_Y, 0)
            est_num = np.where(est_Y <= 1, est_Y, 1)
            F = np.array(pd.read_csv('real-data/F.csv', index_col=0))

            for method in methods:
                if method == 'est':
                    inf, sup, RR, v_inf, v_sup, v_RR = est_bound(po_num=est_num, direct=direct, natural=natural)
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

        if natural:
            if direct:
                res.to_csv('real-data/results/' + NAME + 'dirNmethods' + '.csv')
            else:
                res.to_csv('real-data/results/' + NAME + 'indNmethods' + '.csv')
        else:
            if direct:
                res.to_csv('real-data/results/' + NAME + 'dirZmethods' + '.csv')
            else:
                res.to_csv('real-data/results/' + NAME + 'indZmethods' + '.csv')


def rename(ida, idb):
    os.rename('real-data/T' + str(ida) + '-' + str(idb) + '.csv', 'real-data/T.csv')
    os.rename('real-data/Y' + str(ida) + '-' + str(idb) + '.csv', 'real-data/Y.csv')
    os.rename('real-data/X' + str(ida) + '-' + str(idb) + '.csv', 'real-data/X.csv')
    os.rename('real-data/f_model' + str(ida) + '-' + str(idb) + '.pkl', 'real-data/f_model.pkl')
    os.rename('real-data/pi_model' + str(ida) + '-' + str(idb) + '.pkl', 'real-data/pi_model.pkl')

def run(ida, idb):
    dt.check_path(real=True)
    rename(ida, idb)

    T = np.array(pd.read_csv('real-data/T.csv', index_col=0))
    n = len(T) - 1

    for i in range(0, 5):
        n_val = int(n * (i + 1) / 5)
        set_global(n_val=n_val, ida=ida, idb=idb)
        methods = ['est', 'ML', 'boots']
        pr_bound(methods=methods, direct=True)
        print(i, 5, 'dir')
        pr_bound(methods=methods, direct=False)
        print(i, 5, 'ind')
    # '''

if __name__ == '__main__':
    # concat_by_time(11, 1)
    # making_T(11, 1)
    # balance_T(11, 1)
    run(11, 7)
    exit(0)
