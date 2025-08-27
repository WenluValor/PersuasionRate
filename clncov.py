import pandas as pd
import numpy as np
import re
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder


def cleanX():
    path = 'D:\\real-data\\id-link\\GESISKG_resources_ssoar_links.ttl'
    # path = 'D:\\real-data\\id-link\\test.csv'
    dt = []
    tmp = {}
    i = 0
    pattern = (r'(csa-pais|csa-sa|bszbw-wao|fis-bildung|csa-ps|fes-bib|csa-assia|gesis-solis|csa-pei|ubk-opac|'
               r'dzi-solit|proquest-pao|csa-ssa|dza-gerolit|iab-litdok|gesis-bib|wzb-bib|springer-soj|gesis-ssoar|'
               r'gesis-smarth)(-[a-zA-Z0-9]+)(-[a-zA-Z0-9]+)*')
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        while i < len(lines):
            line = lines[i]
            if line.isspace():
                tmp = {}
            else:
                line = line.replace('\n', '')
                line = line.replace('\t', '')
                if line.startswith('<'):
                    aa = re.findall(pattern, line)
                    if len(aa) == 0:
                        while not line.endswith('.'):
                            i += 1
                            line = lines[i]
                            line = line.replace('\n', '')
                            line = line.replace('\t', '')
                        i += 1
                        continue
                    name = aa[-1][0] + aa[-1][1] + aa[-1][2]
                    tmp['id'] = name

                    pos = line.index('>')
                    pos1 = line.find(':', pos + 1)
                    pos2 = line.find(' ', pos1 + 1)
                    tmp['type'] = line[pos1 + 1: pos2]
                else:
                    if 'gesiskg:linkContext' in line:
                        pos1 = line.find('"')
                        pos2 = line.find('"', pos1 + 1)
                        value = line[pos1 + 1: pos2]
                        tmp['linkContext'] = value
                    elif 'gesiskg:linkScore' in line:
                        pos1 = line.find('"')
                        pos2 = line.find('"', pos1 + 1)
                        value = line[pos1 + 1: pos2]
                        tmp['linkScore'] = value
                    elif 'gesiskg:toSource' in line:
                        pos1 = line.find('"')
                        pos2 = line.find('"', pos1 + 1)
                        value = line[pos1 + 1: pos2]
                        tmp['toSource'] = value
                    elif 'schema:name' in line:
                        pos1 = line.find('"')
                        pos2 = line.find('"', pos1 + 1)
                        value = line[pos1 + 1: pos2]
                        tmp['schemaName'] = value
                if line.endswith('.'):
                    dt.append(tmp)
            i += 1
            # print(i)
    dt = pd.DataFrame(dt)
    # dt.dropna(axis=0, how='any', inplace=True)
    dt.to_csv('D:\\real-data\\id-data\\ssoar-link.csv')
    # dt.to_csv('D:\\real-data\\id-link\\test-output.csv')
    return

def clean_X1():
    id_list = pd.read_csv('D:\\real-data\\id-data\\ex_id.csv')
    for id in range(0, 13):
        path = 'D:\\real-data\\id-data\\ssoar-' + str(id) + '.csv'
        dt = pd.read_csv(path)
        new_dt = dt[dt['id'].isin(list(id_list['ex_id'])) == True]
        new_dt = pd.DataFrame(new_dt)
        new_dt.to_csv('D:\\real-data\\id-data1\\ssoar-' + str(id) + '.csv')

def clean_X2():
    dt = pd.read_csv('D:\\real-data\\id-data1\\ssoar-0.csv')
    for id in range(1, 13):
        path = 'D:\\real-data\\id-data1\\ssoar-' + str(id) + '.csv'
        new_dt = pd.read_csv(path)
        dt = pd.concat([dt, new_dt])
    dt.drop('Unnamed: 0', axis=1, inplace=True)
    dt.drop('Unnamed: 0.1', axis=1, inplace=True)
    dt.drop_duplicates(inplace=True)
    dt.dropna(thresh=3, inplace=True)
    dt.to_csv('D:\\real-data\\id-data2\\ssoar.csv')

def clean_X3():
    dt = pd.read_csv('D:\\real-data\\id-data2\\ssoar.csv')
    df = dt.dropna()
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df.to_csv('D:\\real-data\\id-data2\\allNA.csv')

def clean_X4():
    dt = pd.read_csv('D:\\real-data\\id-data2\\allNA.csv')
    typelist = list(pd.unique(dt['type']))
    for i in range(0, len(dt)):
        if dt.loc[i, 'toSource'] == 'gesis':
            dt.loc[i, 'toSource'] = 0
        else:
            dt.loc[i, 'toSource'] = 1
        dt.loc[i, 'type'] = typelist.index(str(dt.loc[i, 'type']))
        # print(i, len(dt))
    dt.drop('Unnamed: 0', axis=1, inplace=True)
    dt.to_csv('D:\\real-data\\id-data2\\out1.csv')
    typelist = pd.DataFrame(typelist)
    typelist.to_csv('D:\\real-data\\id-data2\\typelist.csv')

def clean_X5():
    dt = pd.read_csv('D:\\real-data\\id-data2\\miniout5.csv')
    vob = []
    for i in range(0, len(dt)):
        s = dt.loc[i, 'schemaName'].split()
        vob.append(s)
    te = TransactionEncoder()
    te_ary = te.fit(vob).transform(vob)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    a = fpgrowth(df, min_support=0.005, use_colnames=True, max_len=1)
    arr = []
    for element in a['itemsets']:
        new = list(element)
        arr.append(new[0])
    a['itemsets'] = arr
    a.to_csv('D:\\real-data\\id-data2\\vobschemaName.csv')
    # print(a['itemsets'])

def clean_X6():
    dt = pd.read_csv('D:\\real-data\\id-data2\\out2.csv')
    mini_dt = dt.sample(frac=0.5, random_state=33)
    # mini_dt['linkContext'] = mini_dt['linkContext'].apply(lambda x: x.lower())
    mini_dt['schemaName'] = mini_dt['schemaName'].apply(lambda x: x.lower())
    mini_dt.drop('Unnamed: 0', axis=1, inplace=True)
    mini_dt.to_csv('D:\\real-data\\id-data2\\miniout5.csv')


def clean_X7():
    dt = pd.read_csv('D:\\real-data\\id-data2\\out2.csv')
    vob = pd.read_csv('D:\\real-data\\id-data2\\vobschemaName.csv')
    vob = list(vob['itemsets'])
    df = []
    for i in range(0, len(dt)):
        words = dt.loc[i, 'schemaName'].lower().split()
        dd = {val: words.count(val) for val in vob}
        df.append(dd)
    df = pd.DataFrame(df)
    df.insert(0, 'id', dt['id'])
    # print(df)
    df.to_csv('D:\\real-data\\id-data2\\out2schemaName.csv')

def clean_X8():
    dt = pd.read_csv('D:\\real-data\\id-data2\\out1.csv')
    dt = dt.drop_duplicates(subset='id')
    dt.to_csv('D:\\real-data\\id-data2\\out2.csv')

if __name__ == '__main__':
    clean_X7()
    exit(0)
