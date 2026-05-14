import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings


def get_prefix(direct, natural):
    if direct & natural:
        aa = 'dirN'
    elif (not direct) & natural:
        aa = 'indN'
    elif direct & (not natural):
        aa = 'dirZ'
    else:
        aa = 'indZ'
    return aa


def get_name(log_n: int, direct, natural):
    n = log_n
    aa = get_prefix(direct=direct, natural=natural)

    name = 'n' + str(n) + aa + 'methods'
    return name


def make_table(ind_size: int, methods):
    for direct in [False, True]:
        for natural in [False, True]:
            col_name = []

            for method in methods:
                col_name.append('inf_' + method)
                col_name.append('sup_' + method)
                col_name.append('RR_' + method)
                col_name.append('rate_cov_' + method)
                col_name.append('v_inf_' + method)
                col_name.append('v_sup_' + method)
                col_name.append('v_RR_' + method)
                col_name.append('rate_inf_' + method)
                col_name.append('rate_sup_' + method)
                col_name.append('rate_RR_' + method)

            num = len(methods) * 10
            ans = pd.DataFrame(np.zeros([ind_size, num]), columns=col_name)

            for i in range(ind_size):
                log_n = int(5 * (i + 2))
                # log_n = 35
                name = get_name(log_n=log_n, direct=direct, natural=natural)
                res = pd.read_csv('test-data/results/' + name + '.csv', index_col=0)
                true_inf = np.array(res.loc[:, 'inf_true'])
                true_sup = np.array(res.loc[:, 'sup_true'])
                true_RR = np.array(res.loc[:, 'RR_true'])

                for method in methods:
                    inf_dt = np.array(res.loc[:, 'inf_' + method])
                    v_inf_dt = np.array(res.loc[:, 'v_inf_' + method])
                    sup_dt = np.array(res.loc[:, 'sup_' + method])
                    v_sup_dt = np.array(res.loc[:, 'v_sup_' + method])
                    RR_dt = np.array(res.loc[:, 'RR_' + method])
                    v_RR_dt = np.array(res.loc[:, 'v_RR_' + method])

                    ans.loc[i, 'inf_' + method] = np.average(inf_dt)
                    ans.loc[i, 'v_inf_' + method] = np.average(v_inf_dt)
                    ans.loc[i, 'sup_' + method] = np.average(sup_dt)
                    ans.loc[i, 'v_sup_' + method] = np.average(v_sup_dt)
                    ans.loc[i, 'RR_' + method] = np.average(RR_dt)
                    ans.loc[i, 'v_RR_' + method] = np.average(v_RR_dt)

                    ans.loc[i, 'rate_inf_' + method] = cov_rate(dt=inf_dt, v=v_inf_dt, true=true_inf, type='inf')
                    ans.loc[i, 'rate_sup_' + method] = cov_rate(dt=sup_dt, v=v_sup_dt, true=true_sup, type='sup')
                    ans.loc[i, 'rate_RR_' + method] = cov_rate(dt=RR_dt, v=v_RR_dt, true=true_RR, type='RR')

            aa = get_prefix(direct=direct, natural=natural)
            ans.to_csv('test-data/sum/' + aa + '.csv')
    return


def cov_rate(dt, v, true, type):
    count = 0
    # '''
    if type == 'RR':
        lower = dt - 1.96 * np.sqrt(v)
        upper = dt + 1.96 * np.sqrt(v)
        for i in range(dt.shape[0]):
            if (true[i] >= lower[i]) & (true[i] <= upper[i]):
                count += 1
    elif type == 'inf':
        # lower = dt - 1.64 * np.sqrt(v)
        lower = dt - 1.96 * np.sqrt(v)
        upper = dt + 1.96 * np.sqrt(v)
        for i in range(dt.shape[0]):
            # if true >= lower[i]:
            if (true[i] >= lower[i]) & (true[i] <= upper[i]):
                count += 1
    else:
        # upper = dt + 1.64 * np.sqrt(v)
        lower = dt - 1.96 * np.sqrt(v)
        upper = dt + 1.96 * np.sqrt(v)
        for i in range(dt.shape[0]):
            # if true <= upper[i]:
            if (true[i] >= lower[i]) & (true[i] <= upper[i]):
                count += 1
    # '''
    '''
    lower = dt - 1.96 * np.sqrt(v)
    upper = dt + 1.96 * np.sqrt(v)
    for i in range(dt.shape[0]):
        if (true >= lower[i]) & (true <= upper[i]):
            count += 1
    '''

    count /= dt.shape[0]
    return count

def make_infsup_plots(ind_size: int, methods, direct, natural, ax, ax2):
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)


    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    styles = ['-', '--', '-.', ':', '--']
    markers = ['o', '^', 's', 'x']
    label_dict = {'est': 'Doubly Robust', 'ora': 'Oracle', 'ML': 'Outcome Model\nRegression', 'IPW': 'IPW'}
    symbol_dict = {'dirZ': r'$\mathcal{L}(\overline{\tau}_{\rm{dpr}, N}(\mathbf{0}))$',
                   'dirN': r'$\mathcal{L}(\overline{\tau}_{\rm{dpr}, N}(\mathbf{T}))$',
                   'indZ': r'$\mathcal{L}(\overline{\tau}_{\rm{ipr}, N}(\omega(\mathbf{0}); \mathbf{0}))$',
                   'indN': r'$\mathcal{L}(\overline{\tau}_{\rm{ipr}, N}(\omega(\mathbf{T}); \mathbf{T}))$'}
    aa = get_prefix(direct=direct, natural=natural)
    x = np.arange(ind_size) * 0.5 + 1
    table = pd.read_csv('test-data/sum/' + aa + '.csv', index_col=0)
    for i in range(len(methods)):
        method = methods[i]
        inf = np.array(table.loc[:, 'inf_' + method])
        true = np.array(table.loc[:, 'inf_true'])
        dt = inf - true

        ax.plot(x, dt, label=label_dict[method], color=colors[i],
                linewidth=1, linestyle=styles[i], marker=markers[i],
                markersize=5)

        '''
        if method == 'boots':
            ax2.plot(x, dt, label=label_dict[method], color=colors[i],
                     linewidth=1, linestyle=styles[i], marker=markers[i],
                    markersize=5)

        else:
            ax.plot(x, dt, label=label_dict[method], color=colors[i],
                    linewidth=1, linestyle=styles[i], marker=markers[i],
                    markersize=5)
        '''

    if (aa == 'indZ') | (aa == 'indN'):
        ax.set_xlabel(r'$\log N$', fontsize=14,
                      fontweight='bold')
        ax.tick_params(axis='x', width=0)
    else:
        ax.tick_params(axis='x', width=0)

    ax.set_ylabel('MSE', fontsize=14,
                  fontweight='bold')
    ax.tick_params(axis='y', direction='inout', color='gray',
                   width=1, length=4)
    ax2.tick_params(axis='y', direction='inout', color='gray',
                    width=0.05, length=0.05)
    ax2.set_yticklabels([])

    '''
    if (aa == 'dirZ') | (aa == 'indZ'):
        ax.set_ylabel('Other Bias', fontsize=14,
                      fontweight='bold')
        ax.tick_params(axis='y', direction='inout', color='gray',
                        width=1, length=4)
        ax2.tick_params(axis='y', direction='inout', color='gray',
                        width=0.05, length=0.05)
        ax2.set_yticklabels([])
    if (aa == 'dirN') | (aa == 'indN'):
        ax2.tick_params(axis='y', direction='inout', color='gray',
                        width=1, length=4)
        ax2.set_ylabel('Boots Bias', fontsize=14,
                       fontweight='bold', labelpad=10)
        ax.tick_params(axis='y', width=0)
    '''

    # ax.set_title(symbol_dict[aa], fontsize=15, fontweight='bold')
    ax.set_title(symbol_dict[aa], fontsize=15)
    ax.set_xticks(x)
    # ax2.grid(linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
    ax.grid(linestyle='--', linewidth=0.5, color='gray', alpha=0.3)


def lineplot():
    methods = ['est', 'ora', 'ML', 'IPW']
    ind_size = 7
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 5),
                                                 sharex='col', sharey='row')
    ax_list = [ax1, ax2, ax3, ax4]
    ax2_list = []
    for i in range(4):
        ax2_list.append(ax_list[i].twinx())
    ax2_list[0].get_shared_y_axes().join(ax2_list[0], ax2_list[1])
    ax2_list[2].get_shared_y_axes().join(ax2_list[2], ax2_list[3])

    for natural in [True, False]:
        for direct in [True, False]:
            i = 2 * (1 - int(direct)) + int(natural)
            make_infsup_plots(ind_size=ind_size, methods=methods,
                              natural=natural, direct=direct, ax=ax_list[i], ax2=ax2_list[i])

    lines1, labels1 = fig.axes[0].get_legend_handles_labels()
    lines2, labels2 = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1, 1))
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.8, top=0.9, wspace=0.05, hspace=0.4)
    plt.savefig('test-data/lineplot.png', dpi=600, bbox_inches='tight')
    # plt.savefig('test-data/lineplot.eps', format='eps', bbox_inches='tight')
    plt.show()


def make_rate_plots(methods):
    fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize=(9, 4), sharey='row')
    ax_list = [ax1, ax2, ax3, ax4]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    label_dict = {'est': 'Doubly Robust', 'ora': 'Oracle', 'ML': 'Outcome Model\nRegression', 'IPW': 'IPW'}
    symbol_dict = {'dirZ inf': r'$\mathcal{L}(\overline{\tau}_{\rm{dpr}, N}(\mathbf{0}))$',
                   'dirN inf': r'$\mathcal{L}(\overline{\tau}_{\rm{dpr}, N}(\mathbf{T}))$',
                   'indZ inf': r'$\mathcal{L}(\overline{\tau}_{\rm{ipr}, N}(\omega(\mathbf{0}); \mathbf{0}))$',
                   'indN inf': r'$\mathcal{L}(\overline{\tau}_{\rm{ipr}, N}(\omega(\mathbf{T}); \mathbf{T}))$',
                   'dirZ sup': r'$\mathcal{U}(\overline{\tau}_{\rm{dpr}, N}(\mathbf{0}))$',
                   'dirN sup': r'$\mathcal{U}(\overline{\tau}_{\rm{dpr}, N}(\mathbf{T}))$',
                   'indZ sup': r'$\mathcal{U}(\overline{\tau}_{\rm{ipr}, N}(\omega(\mathbf{0}); \mathbf{0}))$',
                   'indN sup': r'$\mathcal{U}(\overline{\tau}_{\rm{ipr}, N}(\omega(\mathbf{T}); \mathbf{T}))$',
                   'dirZ RR': r'$\overline{RR}_{\rm{dpr}, N}(\mathbf{0})$',
                   'dirN RR': r'$\overline{RR}_{\rm{dpr}, N}(\mathbf{T})$',
                   'indZ RR': r'$\overline{RR}_{\rm{ipr}, N}(\omega(\mathbf{0}); \mathbf{0})$',
                   'indN RR': r'$\overline{RR}_{\rm{ipr}, N}(\omega(\mathbf{T}); \mathbf{T})$',
                   }

    for i in range(len(methods)):
        categories = []
        values = []
        method = methods[i]
        for type in ['RR', 'sup', 'inf']:
            for direct in [False, True]:
                for natural in [True, False]:
                    aa = get_prefix(direct=direct, natural=natural)
                    table = pd.read_csv('test-data/sum/' + aa + '.csv', index_col=0)

                    val = np.array(table.loc[:, 'rate_' + type + '_' + method])[-1]
                    categories.append(symbol_dict[aa + ' ' + type])
                    values.append(val)
        ax = ax_list[i]
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.barh(categories, values, color=colors[i], label=label_dict[method],
                height=0.7)
        ax.axvline(x=0.95, linestyle='--', color='red', linewidth=0.5)
        ax.grid(linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
        ax.tick_params(axis='y', width=0)
        ax.tick_params(axis='x', width=0)
        ax.set_xlabel('Coverage Rate', fontsize=10)

    fig.legend()


def barplot():
    methods = ['est', 'ora', 'ML', 'IPW']

    make_rate_plots(methods)
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.8, top=0.9, wspace=0.15, hspace=0.4)
    plt.savefig('test-data/barplot.png', dpi=600, bbox_inches='tight')
    # plt.savefig('test-data/barplot.eps', format='eps', bbox_inches='tight')
    plt.show()


def check_path():
    paths = ['test-data/sum']
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    check_path()
    methods = ['true', 'est', 'ora', 'IPW', 'ML']
    # methods = ['est', 'ora', 'ML']
    ind_size = 7
    make_table(ind_size=ind_size, methods=methods)
    lineplot()
    barplot()

    exit(0)
