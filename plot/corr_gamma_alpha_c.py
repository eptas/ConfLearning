import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

cwd = Path.cwd()
path_data = os.path.join(cwd, '../results/fittingData')

models = [2, 3, 4, 5]
nsubjects = 66

model_name = ['Rescorla', 'Rescorla\nZero', 'Rescorla\nConf', 'Rescorla\nConfGen', 'Rescorla\nConfBase', 'Rescorla\nConfBaseGen',
              'Rescorla\nConfZero', 'Rescorla\nConfZeroGen', 'Rescorla\nConfBaseZero', 'Rescorla\nConfBaseZeroGen', 'BayesModel', 'Bayes\nIdealObserver']

fig, axes = plt.subplots(2, 3)
rows, columns = [*[0 for _ in range(3)], *[1 for _ in range(3)]], [0, 1, 2, 0, 1, 2]

for i, m in enumerate(models):

    fittingData = pd.read_pickle(os.path.join(path_data, 'fittingDataM' + str(m) + '.pkl'))

    row, col = rows[m], columns[m]

    corr = np.corrcoef(fittingData.ALPHA_C, fittingData.GAMMA)[0][1]
    axes[row, col].scatter(fittingData.ALPHA_C, fittingData.GAMMA, s=8, c='g', marker='o')
    axes[row, col].set_title(model_name[m])
    axes[1, col].set_xlabel('gamma')
    axes[row, 0].set_ylabel('alpha_c')
    axes[row, col].set_xticks(np.arange(0, 1.2, step=0.2))
    axes[row, col].set_yticks(np.arange(0, 1.2, step=0.2))
    axes[row, col].text(0.5 if corr >= 0 else 0.5, 0.5, str(round(corr, 2)), color='k', fontsize=10)
    axes[row, col].grid('silver', linestyle='-', linewidth=0.4)

fig.savefig('../figures/param_corr/corr_gamma_alpha_c.png', bbox_inches='tight')
plt.close()
