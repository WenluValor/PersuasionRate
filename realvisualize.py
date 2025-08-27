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

def get_real_name(i: int, n, direct, natural, ida, idb):
    aa = get_prefix(direct=direct, natural=natural)

    N = int(n * i / 5)
    name = 'N' + str(N) + 'alg' + str(ida) + '-' + str(idb)
    name1 = name + aa + 'methods'
    return name1


def make_real_table(n, methods, ida, idb):
    ind_size = 5
    for direct in [False, True]:
        for natural in [False, True]:
            col_name = []

            for method in methods:
                col_name.append('inf_' + method)
                col_name.append('sup_' + method)
                col_name.append('RR_' + method)
                col_name.append('v_inf_' + method)
                col_name.append('v_sup_' + method)
                col_name.append('v_RR_' + method)

            num = len(methods) * 6
            ans = pd.DataFrame(np.zeros([ind_size, num]), columns=col_name)

            for i in range(5):
                name1 = get_real_name(i=i + 1, n=n, direct=direct, natural=natural, ida=ida, idb=idb)
                res = pd.read_csv('real-data/results/' + name1 + '.csv', index_col=0)

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

            aa = get_prefix(direct=direct, natural=natural)
            ans.to_csv('real-data/sum/alg' + str(ida) + '-' + str(idb) + aa + '.csv')
    return


def make_infsup_plots(ind_size: int, ida, idb, type, methods, direct, natural, ax, ax2):
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)


    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    styles = ['-', '-.', ':', '--']
    markers = ['o', 's', 'x']
    label_dict = {'est': 'Doubly Robust', 'ML': 'Outcome Model\nRegression', 'boots': 'Bootstrap'}
    coef_dict = {'inf': 1.64, 'sup': 1.64, 'RR': 1.96}
    if type == 'inf':
        symbol_dict = {'dirZ': r'$\mathcal{L}(\overline{\tau}_{\rm{dpr}, N}(\mathbf{0}))$',
                   'dirN': r'$\mathcal{L}(\overline{\tau}_{\rm{dpr}, N}(\mathbf{T}))$',
                   'indZ': r'$\mathcal{L}(\overline{\tau}_{\rm{ipr}, N}(\omega(\mathbf{0}); \mathbf{0}))$',
                   'indN': r'$\mathcal{L}(\overline{\tau}_{\rm{ipr}, N}(\omega(\mathbf{T}); \mathbf{T}))$'}
    elif type == 'sup':
        symbol_dict = {'dirZ': r'$\mathcal{U}(\overline{\tau}_{\rm{dpr}, N}(\mathbf{0}))$',
                       'dirN': r'$\mathcal{U}(\overline{\tau}_{\rm{dpr}, N}(\mathbf{T}))$',
                       'indZ': r'$\mathcal{U}(\overline{\tau}_{\rm{ipr}, N}(\omega(\mathbf{0}); \mathbf{0}))$',
                       'indN': r'$\mathcal{U}(\overline{\tau}_{\rm{ipr}, N}(\omega(\mathbf{T}); \mathbf{T}))$'}
    elif type == 'RR':
        symbol_dict = {'dirZ': r'$\overline{RR}_{\rm{dpr}, N}(\mathbf{0})$',
                       'dirN': r'$\overline{RR}_{\rm{dpr}, N}(\mathbf{T})$',
                       'indZ': r'$\overline{RR}_{\rm{ipr}, N}(\omega(\mathbf{0}); \mathbf{0})$',
                       'indN': r'$\overline{RR}_{\rm{ipr}, N}(\omega(\mathbf{T}); \mathbf{T})$'}
    aa = get_prefix(direct=direct, natural=natural)
    x = np.arange(ind_size) * 0.2 + 0.2
    table = pd.read_csv('real-data/sum/' + 'alg' + str(ida) + '-' + str(idb)
                                         + aa + '.csv', index_col=0)

    for i in range(len(methods)):
        method = methods[i]
        dt = np.array(table.loc[:, type + '_' + method])
        v_dt = np.array(table.loc[:, 'v_' + type + '_' + method])
        coef = coef_dict[type]

        h = 0.02
        left = x - h / 2
        right = x + h / 2
        top = dt + coef * np.sqrt(v_dt)
        bottom = dt - coef * np.sqrt(v_dt)

        if method == 'boots':
            ax2.plot(x, dt, label=label_dict[method], color=colors[i],
                     linewidth=1, linestyle=styles[i], marker=markers[i],
                     markersize=5)
            if type == 'inf':
                ax2.plot([x, x], [dt, bottom], color=colors[i], linewidth=1)
                ax2.plot([left, right], [bottom, bottom], color=colors[i], linewidth=1)
            elif type == 'sup':
                ax2.plot([x, x], [dt, top], color=colors[i], linewidth=1)
                ax2.plot([left, right], [top, top], color=colors[i], linewidth=1)
            elif type == 'RR':
                ax2.plot([x, x], [top, bottom], color=colors[i], linewidth=1)
                ax2.plot([left, right], [bottom, bottom], color=colors[i], linewidth=1)
                ax2.plot([left, right], [top, top], color=colors[i], linewidth=1)

        else:
            ax.plot(x, dt, label=label_dict[method], color=colors[i],
                    linewidth=1, linestyle=styles[i], marker=markers[i],
                    markersize=5)
            if type == 'inf':
                ax.plot([x, x], [dt, bottom], color=colors[i], linewidth=1)
                ax.plot([left, right], [bottom, bottom], color=colors[i], linewidth=1)
            elif type == 'sup':
                ax.plot([x, x], [dt, top], color=colors[i], linewidth=1)
                ax.plot([left, right], [top, top], color=colors[i], linewidth=1)
            elif type == 'RR':
                ax.plot([x, x], [top, bottom], color=colors[i], linewidth=1)
                ax.plot([left, right], [bottom, bottom], color=colors[i], linewidth=1)
                ax.plot([left, right], [top, top], color=colors[i], linewidth=1)

    if (aa == 'indZ') | (aa == 'indN'):
        ax.set_xlabel('Sample Ratio', fontsize=14,
                      fontweight='bold')
        ax.tick_params(axis='x', width=0)
    else:
        ax.tick_params(axis='x', width=0)

    if (aa == 'dirZ') | (aa == 'indZ'):
        ax.set_ylabel('Other Methods', fontsize=14,
                      fontweight='bold')
        ax.tick_params(axis='y', direction='inout', color='gray',
                        width=1, length=4)
        ax2.tick_params(axis='y', direction='inout', color='gray',
                        width=0.05, length=0.05)
        ax2.set_yticklabels([])
    if (aa == 'dirN') | (aa == 'indN'):
        ax2.tick_params(axis='y', direction='inout', color='gray',
                        width=1, length=4)
        ax2.set_ylabel('Boots', fontsize=14,
                       fontweight='bold', labelpad=10)
        ax.tick_params(axis='y', width=0)

    # ax.set_title(symbol_dict[aa], fontsize=15, fontweight='bold')
    ax.set_title(symbol_dict[aa], fontsize=15)
    ax.set_xticks(x)
    ax2.grid(linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
    ax.grid(linestyle='--', linewidth=0.5, color='gray', alpha=0.7)


def lineplot(ida, idb, type):
    methods = ['est', 'ML', 'boots']
    ind_size = 5
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
            make_infsup_plots(ind_size=ind_size, ida=ida, idb=idb, methods=methods, type=type,
                              natural=natural, direct=direct, ax=ax_list[i], ax2=ax2_list[i])

    lines1, labels1 = fig.axes[0].get_legend_handles_labels()
    lines2, labels2 = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1, 1))
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.8, top=0.9, wspace=0.05, hspace=0.4)
    plt.savefig('real-data/' + type + 'lineplot.png', dpi=600, bbox_inches='tight')
    plt.show()


def check_path():
    paths = ['real-data/sum']
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    #check_path()
    #methods = ['est', 'boots', 'ML']
    #ind_size = 5
    # make_real_table(n=39019, methods=methods, ida=11, idb=7)
    # 11-1 174433; 11-4 115425; 11-7 39019

    lineplot(ida=11, idb=7, type='sup')
    lineplot(ida=11, idb=7, type='inf')
    lineplot(ida=11, idb=7, type='RR')
    exit(0)