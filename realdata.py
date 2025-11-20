import pandas as pd
from sklearn.utils import resample
import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.vq import kmeans, whiten
import statsmodels.api as sm

def data_sampled(n_samples, dataset, seed):
    X = dataset[['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7',
                 'f8', 'f9', 'f10', 'f11', 'treatment', 'exposure', 'visit']]
    stratify_cols = pd.concat([dataset['exposure'], dataset['visit']], axis=1)

    X_sampled = resample(
        X,
        n_samples=n_samples,
        stratify=stratify_cols,
        replace=False,
        random_state=seed
    )

    whitened = whiten(X_sampled)
    n_clusters = 2
    kmeaned = kmeans(whitened, n_clusters, seed=seed)
    X = X_sampled

    belongs = []
    for sample in whitened:
        distances = np.linalg.norm(sample - kmeaned[0], axis=1)
        belongs.append(np.argmin(distances))
    X.insert(X.shape[1], 'cluster', belongs)

    for i in range(n_clusters):
        tmp = X[X['cluster'] == i]
        DF = pd.DataFrame(tmp)
        DF.to_csv('real-data/' + str(i) + 'cluster.csv')

    # DF = pd.DataFrame(X_sampled)
    # DF.to_csv('real-data/sampled.csv')

def data_rearrange(dataset):
    scaler = MinMaxScaler()
    feature = dataset[['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7',
                 'f8', 'f9', 'f10', 'f11']]
    scaler.fit(feature)
    feature_scaled = scaler.transform(feature)
    TY = dataset[['exposure', 'visit']] # exposure / treatment
    dataset = np.concatenate([feature_scaled, TY], axis=1)
    distances = np.linalg.norm(feature_scaled - feature_scaled[0], axis=1)
    sorted_indices = np.argsort(distances)
    X_sorted = dataset[sorted_indices]

    DF = pd.DataFrame(X_sorted)
    DF.to_csv('real-data/sorted.csv')

def make_XYT(dataset):
    # print(dataset[0: 10])
    X = dataset[:, 0: 12]
    T = dataset[:, 12]
    Y = dataset[:, 13]
    DF = pd.DataFrame(X)
    DF.to_csv('real-data/X.csv')
    DF = pd.DataFrame(T)
    DF.to_csv('real-data/T.csv')
    DF = pd.DataFrame(Y)
    DF.to_csv('real-data/Y.csv')


if __name__ == '__main__':
    # dataset = pd.read_csv('real-data/criteo-uplift.csv')
    # data_sampled(n_samples=80000, dataset=dataset, seed=2025)

    dataset = pd.read_csv('real-data/0cluster.csv')
    data_rearrange(dataset=dataset)

    dataset = np.array(pd.read_csv('real-data/sorted.csv', index_col=0))
    make_XYT(dataset=dataset)
    exit(0)
