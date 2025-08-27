import numpy as np
import pandas as pd

class Obj():
    def __init__(self):
        self.inf_se = 0
        self.sup_se = 0
        self.rr_se = 0
        self.dir = True
        self.natural = True
        self.logn = 0
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

    def set_numbers(self, case):
        if case == 'simu':
            path_dic = ['p5/r1/results', 'p50/r1/results', 'p500/r1/results']
            path = '/Users/xu/PycharmProjects/InterCausal/interCausal-backup/simu-data/'
            path += path_dic[int(self.param)]
            aa = self.get_prefix()
            name = '/n' + str(self.logn * 10) + 'r1' + aa + 'methods.csv'
            path += name
        elif case == 'hmm':
            path_dic = ['U1T6/results', 'U5T6/results', 'U9T6/results']
            path = '/Users/xu/PycharmProjects/InterCausal/interCausal-backup/hmm-data/'
            path += path_dic[int(self.param)]
            aa = self.get_prefix()
            name = '/n' + str(self.logn * 10) + 'r0' + aa + 'methods.csv'
            path += name
        elif case == 'real':
            path_dic = ['results11-1', 'results11-4', 'results11-7']
            path = '/Users/xu/PycharmProjects/InterCausal/interCausal-backup/real-data/'
            path += path_dic[int(self.param)]
            aa = self.get_prefix()
            # 11-1 174433; 11-4 115425; 11-7 39019
            num_list = [174433, 115425, 39019]
            alg_list = ['alg11-1', 'alg11-4', 'alg11-7']
            num = int(num_list[int(self.param)] * self.logn / 5)
            name = '/N' + str(num) + alg_list[int(self.param)] + aa + 'methods.csv'
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


def create_list(natural, case):
    obj_list = []
    for i in range(54):
        obj = Obj()
        obj.natural = natural
        if i % 3 == 0:
            obj.method = 'est'
        elif i % 3 == 1:
            obj.method = 'ora'
            if case == 'real':
                obj.method = 'boots'
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
    ans = np.zeros([36, 9])
    for i in range(18):
        for j in range(3):
            obj = obj_list[i * 3 + j]
            row = 2 * i
            col = 3 * j
            ans[row, col] = obj.inf
            ans[row, col + 1] = obj.sup
            ans[row, col + 2] = obj.rr
            ans[row + 1, col] = obj.inf_se
            ans[row + 1, col + 1] = obj.sup_se
            ans[row + 1, col + 2] = obj.rr_se


    #DF = pd.DataFrame(ans)
    if natural:
        name = 'N'
    else:
        name = 'Z'
    #DF.to_csv(case + name + '.csv')
    for i in range(36):
        res = ''
        for j in range(3):
            rounded0 = "{:.2f}".format(ans[i, 3 * j])
            rounded1 = "{:.2f}".format(ans[i, 3 * j + 1])
            rounded2 = "{:.2f}".format(ans[i, 3 * j + 2])
            if i % 2 == 0:
                res += ' & '
                res += str(rounded0) + ',' + str(rounded1)
                res += ' & '
                res += str(rounded2)
            else:
                res += ' & '
                res += '(' + str(rounded0) + ',' + str(rounded1) + ')'
                res += ' & '
                res += '(' + str(rounded2) + ')'
        if i % 4 == 0:
            print(i)
            print(res)
            if case == 'real':
                print('\\\\ &   &  ')
            else:
                print('\\\\ &    &       &    &   &  ')
        if i % 4 == 1:
            print(res)
            if case == 'real':
                print('\\\\ \cline{4-9} &  & \multirow{2}{*}{$\\theta_{\ipr}()$}')
            else:
                print('\\\\ \cline{4-12} &  & \multirow{2}{*}{$\\theta_{\ipr}()$}  & \multirow{2}{*}{C} & '
                      '\multirow{2}{*}{a, b} & \multirow{2}{*}{e} ')
        if i % 4 == 2:
            print(res)
            if case == 'real':
                print('\\\\ &   &  ')
            else:
                print('\\\\ &    &       &    &   &  ')
        if i % 4 == 3:
            print(res)

def create_line():
    input_str = input("Enter A, B, C, D (separated by tabs or spaces): ")
    A, B, C, D = map(str.strip, input_str.split())  # Split and strip whitespace

    # Generate LaTeX \multirow code
    latex_code = f"& \\multirow{{2}}{{*}}{{{A}}} & \\multirow{{2}}{{*}}{{{B}, {C}}} & \\multirow{{2}}{{*}}{{{D}}}"

    print("\nGenerated LaTeX code:")
    print(latex_code)

if __name__ == '__main__':
    create_table(natural=True, case='real')
    # create_line()

    exit(0)