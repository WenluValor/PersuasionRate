import numpy as np


def testetst():
    np.random.seed(2026)
    X = np.random.normal(loc=0, scale=1, size=(5, 1))
    print(X)

if __name__ == '__main__':
    for i in range(0, 3):
        testetst()
        print('--------')
    exit(0)