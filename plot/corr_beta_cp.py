import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

n_models = 12

cwd = Path.cwd()
path_data_c = os.path.join(cwd, '../results/choiceProbab')
path_data_f = os.path.join(cwd, '../results/fittingData')

colors = ['r', 'b', 'g', 'y', 'm', 'c']
model_name = ['Rescorla', 'Rescorla\nZero', 'Rescorla\nConf', 'Rescorla\nConfGen', 'Rescorla\nConfBase', 'Rescorla\nConfBaseGen',
              'Rescorla\nConfZero', 'Rescorla\nConfZeroGen', 'Rescorla\nConfBaseZero', 'Rescorla\nConfBaseZeroGen', 'BayesModel', 'Bayes\nIdealObserver']

fig, axes = plt.subplots(4, 3, figsize=(20, 10))
rows, columns = [*[0 for _ in range(3)], *[1 for _ in range(3)], *[2 for _ in range(3)], *[3 for _ in range(3)]], [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

for model in range(n_models):

    choiceProbab = pd.read_pickle(os.path.join(path_data_c, 'choiceProbabM' + str(model) + '.pkl'))
    fittingData = pd.read_pickle(os.path.join(path_data_f, 'fittingDataM' + str(model) + '.pkl'))

    cp_subject = np.mean(choiceProbab[~np.isnan(choiceProbab)], axis=0)

    row, col = rows[model], columns[model]

    corr = np.corrcoef(fittingData.BETA, cp_subject)[0][1]
    axes[row, col].scatter(fittingData.BETA, cp_subject, s=8, c='g', marker='o')
    axes[row, col].set_title(model_name[model])
    axes[max(rows), col].set_xlabel('beta')
    axes[row, 0].set_ylabel('choice probability')
    axes[row, col].set_xticks(np.arange(0, 2.1, step=0.2))
    axes[row, col].set_yticks(np.arange(0.5, 0.75, step=0.1))
    axes[row, col].text(1 if corr >= 0 else 1, 0.5, str(round(corr, 2)), color='k', fontsize=10)
    axes[row, col].grid('silver', linestyle='-', linewidth=0.4)
fig.savefig('../figures/param_corr/corr_beta_cp.png', bbox_inches='tight')
plt.close()
