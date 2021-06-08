import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats

cwd = Path.cwd()
path_data = os.path.join(cwd, '../results/fittingData')

models = np.arange(2, 10)    # CHANGE HERE
nsubjects = 66

model_name = ['Rescorla\nConf', 'Rescorla\nConfGen', 'Rescorla\nConfBase', 'Rescorla\nConfBaseGen',
              'Rescorla\nConfZero', 'Rescorla\nConfZeroGen', 'Rescorla\nConfBaseZero', 'Rescorla\nConfBaseZeroGen']

fig, axes = plt.subplots(3, 3, figsize=(20, 10))
rows, columns = [*[0 for _ in range(3)], *[1 for _ in range(3)], *[2 for _ in range(3)]], [0, 1, 2, 0, 1, 2, 0, 1, 2]

for i, m in enumerate(models):

    fittingData = pd.read_pickle(os.path.join(path_data, 'fittingDataM' + str(m) + '.pkl'))

    row, col = rows[i], columns[i]

    rho, pval = stats.spearmanr(fittingData.ALPHA_C, fittingData.ALPHA)
    # corr = np.corrcoef(fittingData.GAMMA, fittingData.ALPHA_C)[0][1]
    axes[row, col].scatter(fittingData.ALPHA_C, fittingData.ALPHA, s=8, c='g', marker='o')
    # axes[row, col].set_title(model_labels[i])
    axes[row, col].set_xlabel('alpha_c')
    axes[row, col].set_ylabel('alpha')
    axes[row, col].set_xticks(np.arange(0, 1.2, step=0.2))
    axes[row, col].set_yticks(np.arange(0, 1.2, step=0.2))
    axes[row, col].text(0.4 if rho >= 0 else 0.4, 0.5, 'rho = ' + str(round(rho, 2)) + ', p = ' + str(round(pval, 2)), color='k', fontsize=10)
    axes[row, col].grid('silver', linestyle='-', linewidth=0.4)

fig.savefig('../figures/param_corr/corr_alpha_c_alpha.png', bbox_inches='tight')
plt.close()
