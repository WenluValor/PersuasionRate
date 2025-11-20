import math
import numpy as np
import pandas as pd
import os


def get_X(p_val: int, n_val: int, case, alpha):
    N = n_val
    P = p_val
    if case == 'simu':
        X = np.random.normal(loc=0, scale=1, size=(N + 1, P))
    elif case == 'test':
        X = np.zeros([N + 1, 2])
    elif case == 'test-hmm':
        X = np.zeros([N + 1, 2])
        U = 0.1 * np.random.binomial(n=1, p=0.3, size=N + 1)
        DF = pd.DataFrame(U)
        DF.to_csv('test-data/U.csv')
    elif case == 'hmm':
        X = np.random.uniform(low=0, high=1, size=(N + 1, P))
        U = np.zeros([N + 1])
        U[0] = np.random.uniform(low=0, high=1, size=1)[0]
        tmp = U[0]
        for i in range(1, N + 1):
            U[i] = U[i - 1] + np.random.normal(loc=0, scale=alpha, size=1)[0]
        U = (np.sin(U) + 1) / 2
        U[0] = tmp
        DF = pd.DataFrame(U)
        DF.to_csv('test-data/U.csv')
        X = (X + U.reshape(-1, 1)) / 2
    else:
        X = np.random.normal(loc=0, scale=1, size=(N + 1, P))
        pass

    DF = pd.DataFrame(X)
    DF.to_csv('test-data/X.csv')


def get_YT(n_val: int, case):
    N = n_val
    X = np.array(pd.read_csv('test-data/X.csv', index_col=0))
    if (case == 'hmm') | (case == 'test-hmm'):
        U = np.array(pd.read_csv('test-data/U.csv', index_col=0))[0: (N + 1)]

    T = np.zeros([N + 1])
    Y = np.zeros([N + 1])

    for i in range(1, N + 1):
        prob_T = pi_true(x=X[i], t=T[i - 1], y=Y[i - 1], case=case)
        T[i] = np.random.binomial(n=1, p=prob_T, size=1)[0]
        if (case == 'hmm') | (case == 'test-hmm'):
            prob_Y = f_true(x=X[i], t=T[i], y=Y[i - 1], case=case, u=U[i])
        else:
            prob_Y = f_true(x=X[i], t=T[i], y=Y[i - 1], case=case)
        Y[i] = np.random.binomial(n=1, p=prob_Y, size=1)[0]

    DF = pd.DataFrame(T)
    DF.to_csv('test-data/T.csv')
    DF = pd.DataFrame(Y)
    DF.to_csv('test-data/Y.csv')


def pi_true(x: np.array, case, t=0, y=0):
    if case == 'simu':
        if len(x.shape) == 1:
            P = x.shape[0]
            z = 1 * y - 1 * t + sum(x) * 5 / P
            return math.exp(-z) / (1 + math.exp(-z))
        else:
            P = x.shape[1]
            z = 1 * y - 1 * t + np.sum(x, axis=1) * 5 / P
            return np.exp(-z) / (1 + np.exp(-z))
    elif (case == 'test') | (case == 'test-hmm'):
        if len(x.shape) == 1:
            return 1 / 3 + 1 / 3 * t
        else:
            tmp = np.zeros([x.shape[0]])
            return 1 / 3 + 1 / 3 * t + tmp
    elif case == 'hmm':
        if len(x.shape) == 1:
            P = x.shape[0]
            z = 1 + 2.5 * t + sum(x) / (2 * P)
            return 1 / z
        else:
            P = x.shape[1]
            z = 1 + 2.5 * t + np.sum(x, axis=1) / (2 * P)
            return 1 / z
    else:
        pass


def f_true(x: np.array, case, t=0, y=0, u=0):
    if case == 'simu':
        if len(x.shape) == 1:
            P = x.shape[0]
            z = -0.5 * y - 1.5 * t + sum(x) / P
            return math.exp(-z) / (1 + math.exp(-z))
        else:
            P = x.shape[1]
            z = -0.5 * y - 1.5 * t + np.sum(x, axis=1) / P
            return np.exp(-z) / (1 + np.exp(-z))
    elif case == 'test':
        if len(x.shape) == 1:
            return 1 / 4 + 1 / 2 * t
        else:
            tmp = np.zeros([x.shape[0]])
            return 1 / 4 + 1 / 2 * t + tmp
    elif case == 'test-hmm':
        if len(x.shape) == 1:
            return 1 / 4 + 1 / 2 * t + u
        else:
            return 1 / 4 + 1 / 2 * t + u
    elif case == 'hmm':
        b3 = 0.9 # 0.1 0.5 0.9
        b1 = (1 - b3) * 0.6
        b4 = (1 - b3) * 0.2
        if len(x.shape) == 1:
            P = x.shape[0]
            b2 = (1 - b3) / P * 0.2
            z = b1 * t + sum(x) * b2 + b3 * u + b4 * y
            return z
        else:
            P = x.shape[1]
            b2 = (1 - b3) / P * 0.2
            z = b1 * t + np.sum(x, axis=1) * b2 + b3 * u.reshape(-1) + b4 * y
            return z
    else:
        pass


def gene_data(p_val: int, n_val: int, case, alpha):
    get_X(p_val=p_val, n_val=n_val, case=case, alpha=alpha)
    get_YT(n_val=n_val, case=case)
    return


def get_validX(p_val: int, m_val: int, case):
    M = m_val
    P = p_val
    if case == 'simu':
        X = np.random.normal(loc=0, scale=1, size=(M, P))
        return X
    elif (case == 'test') | (case == 'test-hmm'):
        X = np.zeros([M, 2])
        return X
    elif case == 'hmm':
        X = np.random.uniform(low=0, high=1, size=(M, P))
        return X
    else:
        pass


def joint_sample_Y(A, B, rho, case):
    if case == 'simu':
        C = rho * np.sqrt((A - np.square(A)) * (B - np.square(B))) + A * B
    elif case == 'test':
        C = 1 / 8
    else:
        C = A * B

    tmp = np.min((A, B), axis=0)
    C = np.where(C <= tmp, C, tmp)
    tmp = np.max((A + B - 1, 0 * A), axis=0)
    C = np.where(C > tmp, C, tmp)
    A = A.reshape(-1, 1)
    B = B.reshape(-1, 1)
    C = C.reshape(-1, 1)
    # prob of (Y0, Y1) = (0, 0), (0, 1), (1, 0), (1, 1)
    prob_Y = np.hstack((1 + C - A - B, A - C, B - C, C))
    prob_Y = np.where(prob_Y >= 0, prob_Y, 0)
    sample_Y = np.array([np.random.choice(a=4, p=p, size=1)[0] for p in prob_Y])

    return sample_Y


def single_sample_Y(A):
    sample_Y = np.array([np.random.binomial(n=1, p=prob, size=1) for prob in A])
    return sample_Y

def get_validYT(n_val: int, m_val: int, p_val: int, case, rho, alpha):
    P = p_val
    N = n_val
    M = m_val
    U0 = np.random.uniform(low=0, high=1, size=M)
    T0 = np.zeros([M])
    # ZY: ZY0 (denom), ZY1 (dir num), ZY2 (ind num)
    # NY: NY0 (denom), NY1 (dir num), NY2 (ind num), NYt (Y(t))
    for i in range(N + 1):
        X = get_validX(p_val=P, m_val=M, case=case)
        ZY = np.zeros([M, 3])
        NY = np.zeros([M, 4])
        if i == 0:
            pass
        else:
            prev_ZY0 = np.array(pd.read_csv('test-data/valid-data/'
                                       + str(i - 1) + 'ZY.csv', index_col=0))[:, 0]
            prev_ZY1 = np.array(pd.read_csv('test-data/valid-data/'
                                           + str(i - 1) + 'ZY.csv', index_col=0))[:, 1]
            prev_NYt = np.array(pd.read_csv('test-data/valid-data/'
                                            + str(i - 1) + 'NY.csv', index_col=0))[:, 3]
            prev_NY1 = np.array(pd.read_csv('test-data/valid-data/'
                                            + str(i - 1) + 'NY.csv', index_col=0))[:, 1]

            if case == 'hmm':
                U1 = np.sin(np.sinh(2 * U0 - 1) + np.random.normal(loc=0, scale=alpha, size=M))
                U1 = (U1 + 1) / 2
                X = (X + U1.reshape(-1, 1)) / 2
                ZA_dir = f_true(x=X, case=case, t=1, y=prev_ZY0, u=U1)
                ZB = f_true(x=X, case=case, t=0, y=prev_ZY0, u=U1)
                ZA_ind = f_true(x=X, case=case, t=0, y=prev_ZY1, u=U1)
                NA_dir = f_true(x=X, case=case, t=1, y=prev_NYt, u=U1)
                NB_dir = f_true(x=X, case=case, t=0, y=prev_NYt, u=U1)
                NA_ind = f_true(x=X, case=case, t=0, y=prev_NY1, u=U1)
                T1 = pi_true(x=X, case=case, t=T0, y=prev_NYt)
                NA_t = f_true(x=X, case=case, t=T1, y=prev_NYt, u=U1)
                U0 = U1
                T0 = T1
            elif case == 'test-hmm':
                U1 = 0.1 * np.random.binomial(n=1, p=0.3, size=M)
                ZA_dir = f_true(x=X, case=case, t=1, y=prev_ZY0, u=U1)
                ZB = f_true(x=X, case=case, t=0, y=prev_ZY0, u=U1)
                ZA_ind = f_true(x=X, case=case, t=0, y=prev_ZY1, u=U1)
                NA_dir = f_true(x=X, case=case, t=1, y=prev_NYt, u=U1)
                NB_dir = f_true(x=X, case=case, t=0, y=prev_NYt, u=U1)
                NA_ind = f_true(x=X, case=case, t=0, y=prev_NY1, u=U1)
                T1 = pi_true(x=X, case=case, t=T0, y=prev_NYt)
                NA_t = f_true(x=X, case=case, t=T1, y=prev_NYt, u=U1)
                U0 = U1
                T0 = T1
            else:
                ZA_dir = f_true(x=X, case=case, t=1, y=prev_ZY0)
                ZB = f_true(x=X, case=case, t=0, y=prev_ZY0)
                ZA_ind = f_true(x=X, case=case, t=0, y=prev_ZY1)
                NA_dir = f_true(x=X, case=case, t=1, y=prev_NYt)
                NB_dir = f_true(x=X, case=case, t=0, y=prev_NYt)
                NA_ind = f_true(x=X, case=case, t=0, y=prev_NY1)
                T1 = pi_true(x=X, case=case, t=T0, y=prev_NYt)
                NA_t = f_true(x=X, case=case, t=T1, y=prev_NYt)
                T0 = T1

            sample_Y = joint_sample_Y(A=ZA_dir, B=ZB, rho=rho, case=case)
            ZY[:, 0] = np.where((sample_Y == 0) | (sample_Y == 1), 0, 1)
            ZY[:, 1] = np.where((sample_Y == 0) | (sample_Y == 2), 0, 1)
            sample_Y = joint_sample_Y(A=ZA_ind, B=ZB, rho=rho, case=case)
            ZY[:, 2] = np.where((sample_Y == 0) | (sample_Y == 2), 0, 1)

            sample_Y = joint_sample_Y(A=NA_dir, B=NB_dir, rho=rho, case=case)
            NY[:, 0] = np.where((sample_Y == 0) | (sample_Y == 1), 0, 1)
            NY[:, 1] = np.where((sample_Y == 0) | (sample_Y == 2), 0, 1)
            NY[:, 2] = single_sample_Y(A=NA_ind).reshape(-1)
            NY[:, 3] = single_sample_Y(A=NA_t).reshape(-1)


        DF = pd.DataFrame(ZY)
        DF.to_csv('test-data/valid-data/' + str(i) + 'ZY.csv')

        DF = pd.DataFrame(NY)
        DF.to_csv('test-data/valid-data/' + str(i) + 'NY.csv')


def gene_valid(p_val: int, n_val: int, m_val: int, case, rho, alpha):
    get_validYT(n_val=n_val, m_val=m_val, case=case, rho=rho, p_val=p_val, alpha=alpha)
    return


def check_path(real=False):
    if real:
        paths = ['real-data', 'real-data/results']
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        paths = ['test-data', 'test-data/results', 'test-data/valid-data']
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)


if __name__ == '__main__':
    p = 500
    n = int(1e5)
    m = 1000
    alpha = 0.1
    case = 'simu'
    gene_data(p_val=p, n_val=n, case=case, alpha=alpha)
    gene_valid(p_val=p, n_val=n, m_val=m, case=case, alpha=alpha, rho=0.1)
    exit(0)


