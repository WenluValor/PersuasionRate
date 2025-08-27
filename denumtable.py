import numpy as np
import pandas as pd

class Obj():
    def __init__(self):
        self.den = 0
        self.num = 0
        self.dir = True
        self.natural = True
        self.logn = 0
        self.param = 0
        self.method = 'none'

    def get_prefix(self):
        if self.dir & self.natural:
            aa = 'dirN'
        elif (not self.dir) & self.natural:
            aa = 'indN'
        elif self.dir & (not self.natural):
            aa = 'dirZ'
        else:
            aa = 'indZ'
        return aa

    def set_numbers(self, case):
        if case == 'simu':
            path_dic = ['p5/r1/denum-results', 'p50/r1/denum-results', 'p500/r1/denum-results']
            path = '/Users/xu/PycharmProjects/InterCausal/interCausal-backup/simu-data/'
            path += path_dic[int(self.param)]
            aa = self.get_prefix()
            name = '/n' + str(self.logn * 10) + 'r1' + aa + 'denum.csv'
            path += name
        elif case == 'hmm':
            path_dic = ['U1T6/denum-results', 'U5T6/denum-results', 'U9T6/denum-results']
            path = '/Users/xu/PycharmProjects/InterCausal/interCausal-backup/hmm-data/'
            path += path_dic[int(self.param)]
            aa = self.get_prefix()
            name = '/n' + str(self.logn * 10) + 'r0' + aa + 'denum.csv'
            path += name
        res = pd.read_csv(path, index_col=0)
        true_num = np.array(res.loc[0, 'true_num'])
        true_denom = np.array(res.loc[0, 'true_denom'])

        num_dt = np.array(res.loc[:, 'num_' + self.method])
        denom_dt = np.array(res.loc[:, 'denom_' + self.method])

        self.num = np.average(np.square(num_dt - true_num)) * 10000
        self.denom = np.average(np.square(denom_dt - true_denom)) * 10000


def create_list(natural, case):
    obj_list = []
    for i in range(54):
        obj = Obj()
        obj.natural = natural
        if i % 3 == 0:
            obj.method = 'est'
        elif i % 3 == 1:
            obj.method = 'ora'
        else:
            obj.method = 'ML'

        if i <= 17:
            obj.logn = 1
        elif (i > 17) & (i <= 35):
            obj.logn = 3
        else:
            obj.logn = 5

        if (i % 6) <= 2:
            obj.dir = True
        else:
            obj.dir = False

        if (i % 18) <= 5:
            obj.param = 0
        elif ((i % 18) > 5) & ((i % 18) <= 11):
            obj.param = 1
        else:
            obj.param = 2

        obj.set_numbers(case=case)
        obj_list.append(obj)
    return obj_list


def create_table(natural, case):
    obj_list = create_list(natural=natural, case=case)
    ans = np.zeros([18, 6])
    for i in range(18):
        for j in range(3):
            obj = obj_list[i * 3 + j]
            row = i
            col = 2 * j
            ans[row, col] = obj.num
            ans[row, col + 1] = obj.denom

    # DF = pd.DataFrame(ans)
    if natural:
        name = 'N'
    else:
        name = 'Z'
    # DF.to_csv(case + name + '.csv')
    for i in range(18):
        res = ''
        for j in range(3):
            rounded0 = "{:.2f}".format(ans[i, 2 * j])
            rounded1 = "{:.2f}".format(ans[i, 2 * j + 1])
            res += ' & '
            res += str(rounded0)
            res += ' & '
            res += str(rounded1)

        if i % 2 == 0:
            print(i)
            print(res)
            print('\\\\ \cline{4-9} &  & {$\\theta_{\ipr}()$}')
        if i % 2 == 1:
            print(res)


if __name__ == '__main__':
    create_table(natural=True, case='hmm')
    exit(0)