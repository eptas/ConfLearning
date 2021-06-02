import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats

cwd = Path.cwd()

path_data_d = os.path.join(cwd, '../data/')
path_data_r = os.path.join(cwd, '../results/fittingData')

df = pd.read_pickle(os.path.join(path_data_d, 'data.pkl'))

model_name = ['Rescorla\nConf', 'Rescorla\nConfGen', 'Rescorla\nConfBase', 'Rescorla\nConfBaseGen',
              'Rescorla\nConfZero', 'Rescorla\nConfZeroGen', 'Rescorla\nConfBaseZero', 'Rescorla\nConfBaseZeroGen']

modellist = np.arange(2, 10)
nsubjects = 66

fig, axes = plt.subplots(4, 2, figsize=(20, 10))
rows, columns = [*[0 for _ in range(2)], *[1 for _ in range(2)], *[2 for _ in range(2)], *[3 for _ in range(2)]], [0, 1, 0, 1, 0, 1, 0, 1]

for m, model in enumerate(modellist):

    fittingData = pd.read_pickle(os.path.join(path_data_r, 'fittingDataM' + str(model) + '.pkl'))
    row, col = rows[m], columns[m]

    value_inc = np.zeros(nsubjects)

    for s in range(nsubjects):

        value_inc[s] = df[(df.subject == s)].absvaluediff.mean()

    rho, pval = stats.spearmanr(fittingData.ALPHA_C, value_inc)
    axes[row, col].scatter(fittingData.ALPHA_C, value_inc, s=8, c='g', marker='o')
    axes[row, col].set_title(model_name[m])
    axes[max(rows), col].set_xlabel('gamma')
    axes[row, 0].set_ylabel('absolute value differences')
    axes[row, col].set_xticks(np.arange(0, 1.2, step=0.2))
    axes[row, col].set_yticks(np.arange(6.0, 7.7, step=0.5))
    axes[row, col].text(0.4, 6.6 if rho >= 0 else 6.6, 'rho = ' + str(round(rho, 2)) + ', p = ' + str(round(pval, 2)), color='k', fontsize=10)
    axes[row, col].grid('silver', linestyle='-', linewidth=0.4)
fig.savefig('../figures/validation/corr_value_inc_gamma.png', bbox_inches='tight')
plt.close()
