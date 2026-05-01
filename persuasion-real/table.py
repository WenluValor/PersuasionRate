import numpy as np
import pandas as pd

class Obj():
    def __init__(self):
        self.inf_se = 0
        self.sup_se = 0
        self.rr_se = 0
        self.dir = True
        self.natural = True
        self.param = 0
        self.method = 'none'
        self.inf = 0
        self.sup = 0
        self.rr = 0

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

    def set_numbers(self):
        path_dic = ['0results', '1results', '2results']
        # path = 'time/'
        path = 'friend/'
        path += path_dic[int(self.param)]
        aa = self.get_prefix()
        name = '/N4' + aa + 'methods.csv'
        path += name

        res = pd.read_csv(path, index_col=0)

        inf_dt = np.array(res.loc[:, 'inf_' + self.method])
        v_inf_dt = np.array(res.loc[:, 'v_inf_' + self.method])
        sup_dt = np.array(res.loc[:, 'sup_' + self.method])
        v_sup_dt = np.array(res.loc[:, 'v_sup_' + self.method])
        RR_dt = np.array(res.loc[:, 'RR_' + self.method])
        v_RR_dt = np.array(res.loc[:, 'v_RR_' + self.method])

        mask = ((v_inf_dt < 10) & (v_sup_dt < 10) & (v_RR_dt < 10))

        self.inf = np.average(inf_dt[mask]) * 100
        self.inf_se = np.average(np.sqrt(v_inf_dt[mask])) * 10000
        self.sup = np.average(sup_dt[mask]) * 100
        self.sup_se = np.average(np.sqrt(v_sup_dt[mask])) * 10000
        self.rr = np.average(RR_dt[mask]) * 100
        self.rr_se = np.average(np.sqrt(v_RR_dt[mask])) * 10000


def create_list(natural):
    obj_list = []
    for i in range(18):
        obj = Obj()
        obj.natural = natural
        if i <= 5:
            obj.method = 'est'
        elif i >= 12:
            obj.method = 'boots'
        else:
            obj.method = 'ML'

        if (i % 6) <= 1:
            obj.param = 0
        elif (i % 6) >= 4:
            obj.param = 2
        else:
            obj.param = 1

        if (i % 6 == 1) | (i % 6 == 3) | (i % 6 == 5):
            obj.dir = False
        else:
            obj.dir = True

        obj.set_numbers()
        obj_list.append(obj)
    return obj_list


def create_table(type, natural):
    obj_list = create_list(natural=natural)
    for i in range(3): # row
        str1 = ''
        str2 = ''
        for j in range(6): #column
            obj = obj_list[i * 6 + j]
            if type == 'inf':
                value = obj.inf
                se = obj.inf_se
            elif type == 'sup':
                value = obj.sup
                se = obj.sup_se
            else:
                value = obj.rr
                se = obj.rr_se
            value = "{:.2f}".format(value)
            se = "{:.2f}".format(se)
            tmp1 = '& ' + value + ' '
            tmp2 = '& (' + se + ') '
            # tmp1 = obj.method
            # tmp2 = str(obj.dir) + ' ' + str(obj.param)
            str1 += tmp1
            str2 += tmp2
        if i == 0:
            print('\multirow{2}{*}{\\textbf{Doubly Robust}} ' + str1 + '\\\\')
            print(str2 + '\\\\ \\cline{2-7} ')
        elif i == 1:
            print('\multirow{2}{*}{\\textbf{Outcome Model}} ' + str1 + '\\\\')
            print(str2 + '\\\\ \\cline{2-7} ')
        elif i == 2:
            print('\multirow{2}{*}{\\textbf{Bootstrap}} ' + str1 + '\\\\')
            print(str2 + '\\\\ \\hline ')

if __name__ == '__main__':
    create_table(type='RR', natural=True)

    exit(0)