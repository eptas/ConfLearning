import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.transforms


# simuFit = pd.read_pickle("fittingData_para_simu_final.pkl")
simuFit = pd.read_pickle("fittingData_para_simu.pkl")
para_list = ['ALPHA', 'BETA', 'ALPH_N', 'GAMMA']

# grid = dict(
#     ALPHA=np.arange(0.02, 1.00001, 0.01),
#     BETA=np.arange(0.04, 2.01, 0.02),
#     ALPH_N=np.arange(0.02, 1.00001, 0.01),
#     # GAMMA=np.arange(0.4, 20.01, 0.2)
#     GAMMA=np.arange(0.2, 10.01, 0.1)
# )
grid = dict(
    ALPHA=np.arange(0.02, 1.00001, 0.01),
    BETA=np.arange(0.025, 0.501, 0.005),
    ALPH_N=np.arange(0.02, 1.00001, 0.01),
    GAMMA=np.arange(0.2, 10.01, 0.1)
)

param_name = dict(
    ALPHA=r'$\alpha$',
    BETA=r'$\beta$',
    ALPH_N=r'$\alpha_c$',
    GAMMA=r'$\gamma$'
)

sim_list = ['alpha_range', 'beta_range', 'alpha_n_range', 'gamma_range']

plt.figure(figsize=(5, 4.5))
plt.figtext(0.5, 0.01, 'Simulated parameter', fontsize=11, ha='center')
plt.figtext(0.03, 0.4, 'Fitted parameter', fontsize=11, rotation=90, ha='center')

axes = [None] * 4
for p, para in enumerate(para_list):

    # x = simuFit.filter(regex=para).filter(regex=('_' + str(p) + '_'))
    df = simuFit.filter(regex=para).filter(regex=('_' + str(p) + '_')).iloc[: , 2+3*(para == 'BETA'):]
    x = df.values.T
    ngrid, niter = x.shape
    simvals = np.repeat(grid[para], niter)


    axes[p] = plt.subplot(2, 2, p + 1)
    plt.scatter(simvals, x.flatten(), s=8, c='w', marker='o', edgecolors='grey')
    plt.axis('square')


    if p == 1:

        plt.xticks(np.arange(0, 0.6, 0.1))
        plt.yticks(np.arange(0, 0.6, 0.1))
        plt.xlim(0, 0.5)
        plt.ylim(0, 0.5)

    elif p==3:
        plt.xlim(0, 10)
        plt.ylim(0, 10)

        plt.xticks(np.arange(0, 10.1, 2.5))
        plt.yticks(np.arange(0, 10.1, 2.5))
    else:
        plt.xlim(0, 1)
        plt.ylim(0, 1)

    plt.plot(plt.xlim(), plt.ylim(), ls="--", c=".3")  # [0, 0], [0, 1], transform=plt.transAxes)

    plt.text(0.06, 0.9, param_name[para], transform=axes[p].transAxes, fontsize=10)
    plt.annotate(param_name[para], xy=(0.06, 0.94), xycoords='axes fraction', bbox=dict(fc=[1, 1, 1], ec=[0.5, 0.5, 0.5], lw=0.5), ha='left')

    # rho, pval = stats.spearmanr(grid[para], df.mean(axis=0).values)
    # plt.plot(grid[para], df.mean(axis=0).values, color='r')

    # if p == 1:
    #     plt.text(1, 0.4, 'rho = ' + str(round(rho, 2)) + ', p = ' + str(round(pval, 3)), color='black', fontsize=10)
    # elif p == 3:
    #     plt.text(2.5, 15, 'rho = ' + str(round(rho, 2)) + ', p = ' + str(round(pval, 3)), color='black', fontsize=10)
    # else:
    #     plt.text(0.5, 0.1, 'rho = ' + str(round(rho, 2)) + ', p = ' + str(round(pval, 3)), color='black', fontsize=10)

plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0.25)

for ax in axes[2:]:
    ax.set_position(matplotlib.transforms.Bbox(ax.get_position() + np.array([[0, 0.02], [0, 0.02]])))

plt.savefig('parameter_recovery.png', bbox_inches='tight')
    # plt.close()
