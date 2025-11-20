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


def get_name(log_n: int, rho, direct, natural):
    n = log_n
    r = int(10 * rho)
    aa = get_prefix(direct=direct, natural=natural)

    name1 = 'n' + str(n) + 'r' + str(r) + aa + 'methods'
    name2 = 'n' + str(n) + 'r' + str(r) + aa + 'true'
    return name1, name2


def make_table(ind_size: int, rho, methods):
    for direct in [False, True]:
        for natural in [False, True]:
            col_name = []
            col_name.append('val_true')
            col_name.append('inf_true')
            col_name.append('sup_true')
            col_name.append('RR_true')

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

            num = len(methods) * 10 + 4
            ans = pd.DataFrame(np.zeros([ind_size, num]), columns=col_name)

            for i in range(ind_size):
                log_n = int(5 * (i + 2))
                name1, name2 = get_name(log_n=log_n, rho=rho, direct=direct, natural=natural)
                res = pd.read_csv('test-data/results/' + name1 + '.csv', index_col=0)
                true = np.array(pd.read_csv('test-data/results/' + name2 + '.csv', index_col=0))[:, 0]
                ans.loc[i, 'val_true'] = true[0]
                ans.loc[i, 'inf_true'] = true[1]
                ans.loc[i, 'sup_true'] = true[2]
                ans.loc[i, 'RR_true'] = true[3]

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

                    ans.loc[i, 'rate_cov_' + method] = rate(inf_dt=inf_dt, sup_dt=sup_dt, true=true[0])
                    ans.loc[i, 'rate_inf_' + method] = cov_rate(dt=inf_dt, v=v_inf_dt, true=true[1], type='inf')
                    ans.loc[i, 'rate_sup_' + method] = cov_rate(dt=sup_dt, v=v_sup_dt, true=true[2], type='sup')
                    ans.loc[i, 'rate_RR_' + method] = cov_rate(dt=RR_dt, v=v_RR_dt, true=true[3], type='RR')

            aa = get_prefix(direct=direct, natural=natural)
            r = int(10 * rho)
            ans.to_csv('test-data/sum/r' + str(r) + aa + '.csv')
    return


def rate(inf_dt, sup_dt, true):
    count = 0
    for i in range(inf_dt.shape[0]):
        if (true >= inf_dt[i]) & (true <= sup_dt[i]):
            count += 1
    count /= inf_dt.shape[0]
    return count


def cov_rate(dt, v, true, type):
    count = 0
    # '''
    if type == 'RR':
        lower = dt - 1.96 * np.sqrt(v)
        upper = dt + 1.96 * np.sqrt(v)
        for i in range(dt.shape[0]):
            if (true >= lower[i]) & (true <= upper[i]):
                count += 1
    elif type == 'inf':
        lower = dt - 1.64 * np.sqrt(v)
        for i in range(dt.shape[0]):
            if true >= lower[i]:
                count += 1
    else:
        upper = dt + 1.64 * np.sqrt(v)
        for i in range(dt.shape[0]):
            if true <= upper[i]:
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

def make_infsup_plots(ind_size: int, rho, methods, direct, natural, ax, ax2):
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
    label_dict = {'est': 'Doubly Robust', 'ora': 'Oracle', 'ML': 'Outcome Model\nRegression', 'boots': 'Bootstrap'}
    symbol_dict = {'dirZ': r'$\mathcal{L}(\overline{\tau}_{\rm{dpr}, N}(\mathbf{0}))$',
                   'dirN': r'$\mathcal{L}(\overline{\tau}_{\rm{dpr}, N}(\mathbf{T}))$',
                   'indZ': r'$\mathcal{L}(\overline{\tau}_{\rm{ipr}, N}(\omega(\mathbf{0}); \mathbf{0}))$',
                   'indN': r'$\mathcal{L}(\overline{\tau}_{\rm{ipr}, N}(\omega(\mathbf{T}); \mathbf{T}))$'}
    aa = get_prefix(direct=direct, natural=natural)
    x = np.arange(ind_size) * 0.5 + 1
    r = int(10 * rho)
    table = pd.read_csv('test-data/sum/' + 'r' + str(r)
                                         + aa + '.csv', index_col=0)
    #x = x[2: ]
    for i in range(len(methods)):
        method = methods[i]
        inf = np.array(table.loc[:, 'inf_' + method])
        true = np.array(table.loc[:, 'inf_true'])
        dt = inf - true

        if method == 'boots':
            ax2.plot(x, dt, label=label_dict[method], color=colors[i],
                     linewidth=1, linestyle=styles[i], marker=markers[i],
                    markersize=5)

        else:
            ax.plot(x, dt, label=label_dict[method], color=colors[i],
                    linewidth=1, linestyle=styles[i], marker=markers[i],
                    markersize=5)

    if (aa == 'indZ') | (aa == 'indN'):
        ax.set_xlabel(r'$\log N$', fontsize=14,
                      fontweight='bold')
        ax.tick_params(axis='x', width=0)
    else:
        ax.tick_params(axis='x', width=0)

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

    # ax.set_title(symbol_dict[aa], fontsize=15, fontweight='bold')
    ax.set_title(symbol_dict[aa], fontsize=15)
    ax.set_xticks(x)
    ax2.grid(linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
    ax.grid(linestyle='--', linewidth=0.5, color='gray', alpha=0.7)


def lineplot(rho):
    methods = ['est', 'ora', 'ML', 'boots']
    rho = rho
    ind_size = 9
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
            make_infsup_plots(ind_size=ind_size, rho=rho, methods=methods,
                              natural=natural, direct=direct, ax=ax_list[i], ax2=ax2_list[i])

    lines1, labels1 = fig.axes[0].get_legend_handles_labels()
    lines2, labels2 = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1, 1))
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.8, top=0.9, wspace=0.05, hspace=0.4)
    # plt.savefig('test-data/lineplot.png', dpi=600, bbox_inches='tight')
    plt.savefig('test-data/lineplot.eps', format='eps', bbox_inches='tight')
    plt.show()


def make_rate_plots(rho, methods):
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(9, 4), sharey='row')
    ax_list = [ax1, ax2, ax3]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    label_dict = {'est': 'Doubly Robust', 'ora': 'Oracle', 'ML': 'Outcome Model\nRegression', 'boots': 'Bootstrap'}
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

    r = int(10 * rho)

    for i in range(len(methods)):
        categories = []
        values = []
        method = methods[i]
        for type in ['RR', 'sup', 'inf']:
            for direct in [False, True]:
                for natural in [True, False]:
                    aa = get_prefix(direct=direct, natural=natural)
                    table = pd.read_csv('test-data/sum/' + 'r' + str(r)
                                    + aa + '.csv', index_col=0)

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


def barplot(rho):
    methods = ['est', 'ora', 'ML']
    rho = rho

    make_rate_plots(rho, methods)
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.8, top=0.9, wspace=0.15, hspace=0.4)
    # plt.savefig('test-data/barplot.png', dpi=600, bbox_inches='tight')
    plt.savefig('test-data/barplot.eps', format='eps', bbox_inches='tight')
    plt.show()


def boxplot(rho):
    methods = ['est', 'ora', 'ML']
    conditions = ['dirZ', 'dirN', 'indZ', 'indN']
    r = int(10 * rho)
    data = {'dirZ': {'est': 0, 'ora': 0, 'ML': 0}, 'dirN': {'est': 0, 'ora': 0, 'ML': 0},
            'indZ': {'est': 0, 'ora': 0, 'ML': 0}, 'indN': {'est': 0, 'ora': 0, 'ML': 0}}
    symbol_dict = {'dirZ': r'$\mathcal{L}(\overline{\tau}_{\rm{dpr}, N}(\mathbf{0}))$',
                   'dirN': r'$\mathcal{L}(\overline{\tau}_{\rm{dpr}, N}(\mathbf{T}))$',
                   'indZ': r'$\mathcal{L}(\overline{\tau}_{\rm{ipr}, N}(\omega; \mathbf{0}))$',
                   'indN': r'$\mathcal{L}(\overline{\tau}_{\rm{ipr}, N}(\omega; \mathbf{T}))$'}

    for method in methods:
        for direct in [True, False]:
            for natural in [True, False]:
                aa = get_prefix(direct=direct, natural=natural)
                table = pd.read_csv('test-data/results/n50' + 'r' + str(r)
                                + aa + 'methods.csv', index_col=0)
                true = np.array(pd.read_csv('test-data/results/n50' + 'r' + str(r)
                                + aa + 'true.csv', index_col=0))[1]
                val = np.array(table.loc[:, 'inf_' + method])
                data[aa][method] = val - true

    all_data_list = []
    positions_list = []

    offset = [-0.24, 0, 0.24]
    group_center = [1, 2, 3, 4]

    for j, method in enumerate(methods):
        all_data = []
        positions = []
        for i, condition in enumerate(conditions):
            all_data.append(data[condition][method])
            positions.append(group_center[i] + offset[j])
        all_data_list.append(all_data)
        positions_list.append(positions)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, method in enumerate(methods):
        flierprops = dict(marker='o', markerfacecolor=colors[i],
                          markersize=5, markeredgecolor='none')
        meanlineprops = dict(linestyle='--', color='red', linewidth=0.5)
        medianlineprops = dict(linestyle='-', color='black', linewidth=0)
        bp = ax.boxplot(all_data_list[i], positions=positions_list[i], notch=True,
                        widths=0.22, patch_artist=True,
                        showmeans=True, meanline=True, flierprops=flierprops,
                        meanprops=meanlineprops, medianprops=medianlineprops,
                        capprops=dict(color=colors[i]), whiskerprops=dict(color=colors[i]))
        for patch in bp['boxes']:
            patch.set_facecolor(colors[i])
            patch.set_edgecolor('none')

    x_ticks = []
    for item in conditions:
        x_ticks.append(symbol_dict[item])

    ax.set_xticks(group_center, conditions)
    ax.set_xticklabels(x_ticks, fontsize=15, fontweight='bold')
    ax.set_ylabel('Bias', fontsize=15, fontweight='bold', labelpad=10)
    ax.grid(linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
    ax.tick_params(axis='x', width=0)
    ax.tick_params(axis='y', width=0)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Doubly Robust'),
        Patch(facecolor='#ff7f0e', label='Oracle'),
        Patch(facecolor='#2ca02c', label='Outcome Model\nRegression')
    ]
    # upper right for hmm, right for last two simu
    fig.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    # plt.savefig('test-data/boxplot.png', dpi=600, bbox_inches='tight')
    plt.savefig('test-data/boxplot.eps', format='eps', bbox_inches='tight')
    plt.show()


def check_path():
    paths = ['test-data/sum']
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # check_path()
    # methods = ['est', 'ora', 'boots', 'ML']
    # ind_size = 9
    # make_table(ind_size=ind_size, rho=0.1, methods=methods)
    lineplot(rho=0.0)
    barplot(rho=0.0)
    boxplot(rho=0.0)

    exit(0)
