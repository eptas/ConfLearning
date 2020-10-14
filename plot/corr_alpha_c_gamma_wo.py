import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats

cwd = Path.cwd()
path_data = os.path.join(cwd, '../results/fittingData')

models = np.arange(12, 14)    # CHANGE HERE
nsubjects = 66

model_name = ['Rescorla\nConfGamma', 'Rescorla\nConfGenGamma']

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
columns = [0, 1]

for i, m in enumerate(models):

    fittingData = pd.read_pickle(os.path.join(path_data, 'fittingDataM' + str(m) + '.pkl'))

    col = columns[i]

    rho, pval = stats.spearmanr(fittingData.ALPHA_C, fittingData.ALPHA_N)
    # corr = np.corrcoef(fittingData.ALPHA_C, fittingData.GAMMA)[0][1]
    axes[col].scatter(fittingData.ALPHA_C, fittingData.ALPHA_N, s=8, c='g', marker='o')
    axes[col].set_title(model_name[i])
    axes[col].set_xlabel('alpha_c')
    axes[0].set_ylabel('gamma_wo')
    axes[col].set_xticks(np.arange(0, 1.2, step=0.2))
    axes[col].set_yticks(np.arange(0, 1.2, step=0.2))
    axes[col].text(0.4 if rho >= 0 else 0.4, 0.5, 'rho = ' + str(round(rho, 2)) + ', p = ' + str(round(pval, 2)), color='k', fontsize=10)
    axes[col].grid('silver', linestyle='-', linewidth=0.4)

fig.savefig('../figures/param_corr/corr_alpha_c_gamma_wo.png', bbox_inches='tight')
plt.close()
