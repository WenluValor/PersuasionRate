import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from poest import setup
import data as dt
from data import f_true
import math

def charu():
    time = np.arange(12)
    time_res = np.interp(np.linspace(0, 1, 20), np.linspace(0, 1, 12), time)
    print(time_res)

if __name__ == '__main__':
    charu()
    exit(0)