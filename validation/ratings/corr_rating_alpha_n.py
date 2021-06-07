import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats

cwd = Path.cwd()

path_data_d = os.path.join(cwd, '../../data/')
path_data_r = os.path.join(cwd, '../../results/fittingData')

df = pd.read_pickle(os.path.join(path_data_d, 'data.pkl'))

model_name = ['Rescorla\nZero', 'Rescorla\nConfZero', 'Rescorla\nConfZeroGen', 'Rescorla\nConfBaseZero', 'Rescorla\nConfBaseZeroGen']

modellist = [1, 6, 7, 8, 9]
nsubjects = 66

fig, axes = plt.subplots(3, 2, figsize=(20, 10))
rows, columns = [0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]

for m, model in enumerate(modellist):

    fittingData = pd.read_pickle(os.path.join(path_data_r, 'fittingDataM' + str(model) + '.pkl'))
    row, col = rows[m], columns[m]

    delta = np.zeros(nsubjects)

    for s in range(nsubjects):

        delta[s] = df[(df.subject == s) & df.type_rating2].rating.mean() - df[(df.subject == s) & df.type_rating1].rating.mean()

    if m == 0:
        fittingData.ALPHA_N = fittingData.GAMMA

    rho, pval = stats.spearmanr(fittingData.ALPHA_N, delta)
    axes[row, col].scatter(fittingData.ALPHA_N, delta, s=8, c='g', marker='o')
    axes[row, col].set_title(model_name[m])
    axes[max(rows), col].set_xlabel('alpha_n')
    axes[row, 0].set_ylabel('normalized value ratings (p2-p1)')
    axes[row, col].set_xticks(np.arange(0, 1.2, step=0.2))
    axes[row, col].set_yticks(np.arange(-0.12, 0.13, step=0.05))
    axes[row, col].text(0.4, 0 if rho >= 0 else 0, 'rho = ' + str(round(rho, 2)) + ', p = ' + str(round(pval, 2)), color='k', fontsize=10)
    axes[row, col].grid('silver', linestyle='-', linewidth=0.4)
fig.savefig('../../figures/validation/ratings/corr_rating_alpha_n.png', bbox_inches='tight')
plt.close()
