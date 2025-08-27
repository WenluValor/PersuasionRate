import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm


def sep_data():
    chunk_size = 10000
    chunks = pd.read_csv('D:\\real-data\\recommendation_log_0.csv', chunksize=chunk_size,
                         sep='\t')

    col_names = []
    for chunk in chunks:
        col_names = chunk.columns.tolist()
        break

    id_dict = {}
    for i in range(1, 6):
        # log_0 is all nan
        chunks = pd.read_csv('D:\\real-data\\recommendation_log_' + str(i) + '.csv',
                             chunksize=chunk_size, sep='\t', names=col_names)
        for chunk in chunks:
            id_set = pd.unique(chunk['algorithm_id'])
            id_set = id_set[~np.isnan(id_set)].astype(int)
            if len(id_set) != 0:
                for id in id_set:
                    path = 'D:\\real-data\\sep-data1\\alg' + str(id)
                    chunk.drop(chunk[np.isnan(chunk['algorithm_id'])].index, inplace=True)
                    chunk['algorithm_id'] = chunk['algorithm_id'].astype(int)
                    dt = chunk[chunk['algorithm_id'] == id]
                    if not os.path.exists(path):
                        os.makedirs(path)
                        id_dict[id] = 0
                    else:
                        id_dict[id] += 1
                    dt.to_csv(path + '\\' + str(id_dict[id]) + '.csv')
        print(i)


def concat_data():
    for id in range(1, 248):
        path1 = 'D:\\real-data\\sep-data1\\alg' + str(id)
        path2 = 'D:\\real-data\\sep-data2\\alg' + str(id)
        file_num = len(os.listdir(path1))
        dt = pd.read_csv(path1 + '\\' + str(0) + '.csv')
        for j in range(1, file_num):
            new_dt = pd.read_csv(path1 + '\\' + str(j) + '.csv')
            dt = pd.concat([dt, new_dt])
            print(id, j)
        dt.to_csv(path2 + '.csv')


def del_data():
    for id in range(1, 248):
        path2 = 'D:\\real-data\\sep-data2\\alg' + str(id) + '.csv'
        path3 = 'D:\\real-data\\sep-data3\\alg' + str(id) + '.csv'
        dt = pd.read_csv(path2)
        dt.drop('Unnamed: 0.1', axis=1, inplace=True)
        dt.drop('Unnamed: 0', axis=1, inplace=True)
        dt.drop('Unnamed: 17', axis=1, inplace=True)
        dt['clicked'] = dt['clicked'].fillna(0).apply(lambda x: 1 if x != 0 else 0)
        dt.to_csv(path3)
        print(id)

def clean_X():
    dt = pd.read_csv('D:\\real-data\external_IDs.csv', sep='\t', skiprows=8673688,
                     names=['docu_id', 'source', 'ex_id'])
    # print(dt[0: 10])
    dt.drop(dt[dt['source'] != 'mendeley'].index, inplace=True)
    # print(dt[0: 10])
    dt.index = np.arange(len(dt))
    dt.to_csv('D:\\real-data\\sep-data2\\ex_id.csv')


def clean_X1():
    dt = pd.read_csv('D:\\real-data\\sep-data1\\ex_id.csv')
    s = dt['ex_id'].str.split('-', expand=True, n=2)
    s1 = s.iloc[:, 0] + '-' + s.iloc[:, 1]
    # print(s1[0: 10])
    # ds = pd.unique(s1)
    # ds = np.concatenate((ds, np.zeros([len(ds), 2])), axis=1)
    a = s1.value_counts()
    b = s1.value_counts(normalize=True)
    c = pd.concat([a, b], axis=1)
    DF = pd.DataFrame(c)
    DF.to_csv('D:\\real-data\\sep-data1\\ex_lib.csv')

def clean_X2():
    dt = pd.read_csv('D:\\real-data\\sep-data1\\ex_lib.csv')
    dt.rename(columns={'Unnamed: 0': 'source'}, inplace=True)
    DF = pd.DataFrame(dt)
    DF.to_csv('D:\\real-data\\sep-data1\\ex_lib1.csv')
    # print(dt)


def clean_Y():
    for id in range(1, 248):
        path3 = 'D:\\real-data\\sep-data3\\alg' + str(id) + '.csv'
        dt = pd.read_csv(path3)
        length = dt.shape[0]
        index = 0
        new_dt = []
        while index < length:
            size = dt['set_size'][index]
            if dt.loc[index: index + size, 'clicked'].any():
                dt.loc[index, 'clicked'] = 1
            tmp = dt.iloc[index].to_dict()
            new_dt.append(tmp)
            index += size
        path4 = 'D:\\real-data\\sep-data4\\alg' + str(id) + '.csv'
        new_dt = pd.DataFrame(new_dt)
        new_dt.drop('Unnamed: 0', axis=1, inplace=True)
        new_dt.drop('rank_in_set', axis=1, inplace=True)
        new_dt.to_csv(path4)
    return


def sep_data2():
    for id in range(1, 248):
        path = 'D:\\real-data\\sep-data4\\alg' + str(id) + '.csv'
        dt = pd.read_csv(path)
        new_dt = dt[['source_document_id', 'request_received', 'clicked', 'algorithm_id']]
        new_dt.to_csv('D:\\real-data\\sep-data5\\alg' + str(id) + '.csv')


def sep_data3():
    id_dt = pd.read_csv('D:\\real-data\\id-data\\ex_id.csv')
    cln_id = pd.read_csv('D:\\real-data\\id-data2\\out2.csv')
    for id in range(1, 248):
        path = 'D:\\real-data\\sep-data5\\alg' + str(id) + '.csv'
        dt = pd.read_csv(path)
        dt.rename(columns={'source_document_id': 'docu_id'}, inplace=True)
        new_dt = pd.merge(dt, id_dt[['docu_id', 'ex_id']], on='docu_id', how='left')
        new_dt.drop('docu_id', axis=1, inplace=True)
        new_dt.drop('Unnamed: 0', axis=1, inplace=True)
        new_dt.insert(loc=0, column='id', value=new_dt['ex_id'])
        new_dt.drop('ex_id', axis=1, inplace=True)
        new_dt = new_dt[new_dt['id'].isin(list(cln_id['id'])) == True]
        new_dt = new_dt.reset_index(drop=True)
        new_dt.to_csv('D:\\real-data\\sep-data6\\alg' + str(id) + '.csv')

def concat_by_time(ida, idb):
    patha = 'D:\\real-data\\sep-data6\\alg' + str(ida) + '.csv'
    pathb = 'D:\\real-data\\sep-data6\\alg' + str(idb) + '.csv'
    dt_a = pd.read_csv(patha)
    dt_b = pd.read_csv(pathb)
    dt = pd.concat([dt_a, dt_b], axis=0)
    dt['request_received'] = pd.to_datetime(dt['request_received'])
    dt.sort_values(by='request_received', ascending=True, inplace=True)
    dt.drop('Unnamed: 0', axis=1, inplace=True)
    dt = dt.reset_index(drop=True)
    dt.to_csv('D:\\real-data\\sep-data7\\alg' + str(ida) + '-' + str(idb) + '.csv')
    return

def making_alg_list():
    new_dt = []
    for id in range(1, 248):
        path = 'D:\\real-data\\sep-data4\\alg' + str(id) + '.csv'
        dt = pd.read_csv(path)
        new_dt.append(dt.iloc[0])
    new_dt = pd.DataFrame(new_dt)
    new_dt.to_csv('D:\\real-data\\sep-data7\\alg_list.csv')

def making_T(ida, idb):
    dt = pd.read_csv('D:\\real-data\\sep-data7\\alg' + str(ida) + '-' + str(idb) + '.csv')
    Y = dt['clicked'].values
    dt['algorithm_id'] = dt['algorithm_id'].fillna(0).apply(lambda x: 1 if x != ida else 0)
    T = dt['algorithm_id'].values
    vob0 = pd.read_csv('D:\\real-data\\id-data3\\out2.csv')
    vob1 = pd.read_csv('D:\\real-data\\id-data3\\out2schemaName.csv')
    # vob2 = pd.read_csv('D:\\real-data\\id-data3\\out2linkContext.csv')
    tmp_dt = dt[['id', 'clicked']]
    tmp_dt.to_csv('tmp_dt.csv')
    X = tmp_dt.merge(vob0, on='id', how='left')[['type', 'linkScore', 'toSource']].values
    vob = [vob1]
    for i in range(1):
        new_X = tmp_dt.merge(vob[i], on='id', how='left')
        new_X.drop('id', axis=1, inplace=True)
        new_X.drop('clicked', axis=1, inplace=True)
        new_X.drop('Unnamed: 0', axis=1, inplace=True)
        new_X = new_X.values
        X = np.append(X, new_X, axis=1)
    X = pd.DataFrame(X)
    X = X.reset_index(drop=True)
    X.to_csv('D:\\real-data\\sep-data7\\X' + str(ida) + '-' + str(idb) + '.csv')
    T = pd.DataFrame(T)
    T = T.reset_index(drop=True)
    T.to_csv('D:\\real-data\\sep-data7\\T' + str(ida) + '-' + str(idb) + '.csv')
    Y = pd.DataFrame(Y)
    Y = Y.reset_index(drop=True)
    Y.to_csv('D:\\real-data\\sep-data7\\Y' + str(ida) + '-' + str(idb) + '.csv')

def balance_T(ida, idb):
    X = pd.read_csv('D:\\real-data\\sep-data7\\X' + str(ida) + '-' + str(idb) + '.csv')
    Y = pd.read_csv('D:\\real-data\\sep-data7\\Y' + str(ida) + '-' + str(idb) + '.csv')
    T = pd.read_csv('D:\\real-data\\sep-data7\\T' + str(ida) + '-' + str(idb) + '.csv')
    T = np.array(T)[:, 1].reshape(-1, 1)
    Y = np.array(Y)[:, 1].reshape(-1, 1)
    X = np.array(X)[:, 1:]

    Y_index = Y.nonzero()[0]
    if len(Y_index) >= len(Y) - len(Y_index):
        pass
    else:
        n = len(Y) - 2 * len(Y_index)
        sample_index = np.random.choice(Y_index, n)
        N = len(Y) - 1
        new_index = np.hstack((np.arange(1, len(Y)), sample_index))
        # print(T.shape, Y.shape, X.shape)
        # data = np.hstack((T[0: N], Y[0: N], X[1: N + 1]))
        # result = np.ravel(T[1: N + 1])
        data = np.hstack((T[new_index - 1], Y[new_index - 1], X[new_index]))
        result = np.ravel(T[new_index])
        Xtrain, Ytrain = shuffle(data, result, random_state=115)

        model = LogisticRegression(penalty='l1', solver='liblinear')
        # model = LogisticRegression()
        model.fit(Xtrain, Ytrain)

        pkl_filename = 'D:\\real-data\\sep-data7\\pi_model' + str(ida) + '-' + str(idb) + '.pkl'
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model, file)

        data = np.hstack((T[new_index], Y[new_index - 1], X[new_index]))
        result = np.ravel(Y[new_index])
        Xtrain, Ytrain = shuffle(data, result, random_state=10)

        model = LogisticRegression(penalty='l1', solver='liblinear')
        model.fit(Xtrain, Ytrain)

        pkl_filename = 'D:\\real-data\\sep-data7\\f_model' + str(ida) + '-' + str(idb) + '.pkl'
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model, file)
        return

def see_acc(ida, idb):
    f_filename = 'D:\\real-data\\sep-data8\\f_model' + str(ida) + '-' + str(idb) + '.pkl'
    with open(f_filename, 'rb') as file:
        f_model = pickle.load(file)
    pi_filename = 'D:\\real-data\\sep-data8\\pi_model' + str(ida) + '-' + str(idb) + '.pkl'
    with open(pi_filename, 'rb') as file:
        pi_model = pickle.load(file)

    X = pd.read_csv('D:\\real-data\\sep-data8\\X' + str(ida) + '-' + str(idb) + '.csv')
    Y = pd.read_csv('D:\\real-data\\sep-data8\\Y' + str(ida) + '-' + str(idb) + '.csv')
    T = pd.read_csv('D:\\real-data\\sep-data8\\T' + str(ida) + '-' + str(idb) + '.csv')
    T = np.array(T)[:, 1].reshape(-1, 1)
    Y = np.array(Y)[:, 1].reshape(-1, 1)
    X = np.array(X)[:, 1:]
    N = len(Y) - 1
    sample_index = np.random.choice(np.arange(1, N), 1000)
    data = np.hstack((T[sample_index - 1], Y[sample_index - 1], X[sample_index]))
    result = np.ravel(T[sample_index])
    acc1 = pi_model.score(data, result)
    data = np.hstack((T[sample_index], Y[sample_index - 1], X[sample_index]))
    result = np.ravel(Y[sample_index])
    acc2 = f_model.score(data, result)
    print(acc1, acc2)

def aug_T(ida, idb):
    X = pd.read_csv('D:\\real-data\\sep-data7\\X' + str(ida) + '-' + str(idb) + '.csv')
    Y = pd.read_csv('D:\\real-data\\sep-data7\\Y' + str(ida) + '-' + str(idb) + '.csv')
    T = pd.read_csv('D:\\real-data\\sep-data7\\T' + str(ida) + '-' + str(idb) + '.csv')
    T = np.array(T)[:, 1].reshape(-1, 1)
    Y = np.array(Y)[:, 1].reshape(-1, 1)
    X = np.array(X)[:, 1:]

    X = MinMaxScaler().fit_transform(X)
    features = np.hstack((X, T))
    sm = SMOTE(k_neighbors=5)
    new_features, new_Y = sm.fit_resample(features, Y)
    m = new_features.shape[1]
    new_X = new_features[:, 0: m - 1]
    new_T = np.round(new_features[:, m - 1])
    DF = pd.DataFrame(new_X)
    DF.to_csv('D:\\real-data\\sep-data8\\X' + str(ida) + '-' + str(idb) + '.csv')
    DF = pd.DataFrame(new_T)
    DF.to_csv('D:\\real-data\\sep-data8\\T' + str(ida) + '-' + str(idb) + '.csv')
    DF = pd.DataFrame(new_Y)
    DF.to_csv('D:\\real-data\\sep-data8\\Y' + str(ida) + '-' + str(idb) + '.csv')

def learn_fg(ida, idb):
    X = pd.read_csv('D:\\real-data\\sep-data8\\X' + str(ida) + '-' + str(idb) + '.csv')
    Y = pd.read_csv('D:\\real-data\\sep-data8\\Y' + str(ida) + '-' + str(idb) + '.csv')
    T = pd.read_csv('D:\\real-data\\sep-data8\\T' + str(ida) + '-' + str(idb) + '.csv')
    T = np.array(T)[:, 1].reshape(-1, 1)
    Y = np.array(Y)[:, 1].reshape(-1, 1)
    X = np.array(X)[:, 1:]

    N = len(Y) - 1
    new_index = np.arange(1, N)

    data = np.hstack((T[new_index], Y[new_index - 1], X[new_index]))
    result = np.ravel(T[new_index])
    Xtrain, Ytrain = shuffle(data, result, random_state=115)

    model = LogisticRegression(penalty='l1', solver='liblinear')
    # model = LogisticRegression()
    model.fit(Xtrain, Ytrain)

    pkl_filename = 'D:\\real-data\\sep-data8\\pi_model' + str(ida) + '-' + str(idb) + '.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

    data = np.hstack((T[new_index], Y[new_index - 1], X[new_index]))
    result = np.ravel(Y[new_index])
    Xtrain, Ytrain = shuffle(data, result, random_state=10)

    model = LogisticRegression(penalty='l1', solver='liblinear')
    model.fit(Xtrain, Ytrain)

    pkl_filename = 'D:\\real-data\\sep-data8\\f_model' + str(ida) + '-' + str(idb) + '.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

def test_Y(ida, idb):
    X = pd.read_csv('D:\\real-data\\sep-data8\\X' + str(ida) + '-' + str(idb) + '.csv')
    Y = pd.read_csv('D:\\real-data\\sep-data8\\Y' + str(ida) + '-' + str(idb) + '.csv')
    T = pd.read_csv('D:\\real-data\\sep-data8\\T' + str(ida) + '-' + str(idb) + '.csv')
    T = np.array(T)[:, 1].reshape(-1, 1)
    Y = np.array(Y)[:, 1].reshape(-1, 1)
    X = np.array(X)[:, 1:]

    # print(np.average(Y[T == 0]))
    # print(np.average(Y[T == 1]))
    N = Y.shape[0] - 1
    X_design = np.column_stack((T[0: N], Y[0: N]))
    X_design = sm.add_constant(X_design)
    Y_design = T[1: N + 1]
    model = sm.Logit(Y_design, X_design)
    # model = sm.OLS(Y_design, X_design)
    results = model.fit()
    print(results.summary())
    print(ida, idb)


if __name__ == '__main__':
    # sep_data()
    # concat_data()
    # del_data()
    # clean_X()
    # clean_Y()
    # sep_data3()
    # making_alg_list()
    # concat_by_time(9, 11)
    # making_T(9, 11)
    # balance_T(34, 51)
    # see_acc(34, 51)
    # aug_T(11, 7)
    # learn_fg(11, 7)
    # see_acc(11, 1)
    test_Y(11, 7)
    exit(0)
