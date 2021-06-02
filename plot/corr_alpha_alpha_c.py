import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats

cwd = Path.cwd()
path_data = os.path.join(cwd, '../results/fittingData')

models = np.arange(1, 10)
n_subjects = 66

model_name = ['Rescorla\nStatic', 'Rescorla\nConf', 'Rescorla\nConfGen', 'Rescorla\nConfBase', 'Rescorla\nConfNofeedGen',
              'Rescorla\nConfZero', 'Rescorla\nConfZeroGen', 'Rescorla\nConfBaseZero', 'Rescorla\nConfBaseZeroGen']

fig, axes = plt.subplots(3, 3, figsize=(20, 10))
rows, columns = [*[0 for _ in range(3)], *[1 for _ in range(3)], *[2 for _ in range(3)]], [0, 1, 2, 0, 1, 2, 0, 1, 2]

for i, m in enumerate(models):

    subj_data = None

    fittingData = pd.read_pickle(os.path.join(path_data, 'fittingDataM' + str(m) + '.pkl'))

    row, col = rows[i], columns[i]

    rho, pval = stats.spearmanr(fittingData.ALPHA_C, fittingData.ALPHA)
    # corr = np.corrcoef(fittingData.ALPHA_C, fittingData.ALPHA)[0][1]
    axes[row, col].scatter(fittingData.ALPHA_C, fittingData.ALPHA, s=8, c='g', marker='o')
    axes[row, col].set_title(model_name[i])
    axes[max(rows), col].set_xlabel('alpha_c (p1)')
    axes[row, col].set_ylabel('alpha (p0)')
    axes[row, col].set_xticks(np.arange(0, 10.2, step=1))
    axes[row, col].set_yticks(np.arange(0, 1.2, step=0.2))
    axes[row, col].text(5 if rho >= 0 else 5, 0.4, 'rho = ' + str(round(rho, 2)) + ', p = ' + str(round(pval, 3)), color='k', fontsize=10)
    axes[row, col].grid('silver', linestyle='-', linewidth=0.4)

# os.makedirs('../figures/param_corr')
fig.savefig('../figures/param_corr/corr_alpha_c_alpha.png', bbox_inches='tight')
plt.close()
