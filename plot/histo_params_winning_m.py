import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

cwd = Path.cwd()
path_data = os.path.join(cwd, '../results/')
# os.makedirs('../figures/fitting')

n = 3
n_subjects = 66


fittingData = pd.read_pickle(os.path.join(path_data, 'fittingData/fittingDataM' + str(n) + '.pkl'))

alpha = fittingData.ALPHA
beta = fittingData.BETA
gamma = fittingData.ALPHA_C
alpha_c = fittingData.GAMMA

param_fit = [alpha, beta, gamma, alpha_c]
param_name = ['alpha', 'beta', 'gamma', 'alpha_c']
model_name = 'Rescorla\nConfGen'

fig, axes = plt.subplots(2, 2, figsize=(20, 10))
rows, columns = [*[0 for _ in range(2)], *[1 for _ in range(2)]], [0, 1, 0, 1]

for p, para in enumerate(param_fit):

    row, col = rows[p], columns[p]

    axes[row, col].hist(para[1:len(para)], bins=20, color='g')
    axes[max(rows), col].set_xlabel(model_name, fontweight='bold')
    # axes[row, col].set_title(param_name[p])
    axes[row, col].set_yticks(np.arange(0, 42, step=5))
    axes[row, col].set_ylabel(param_name[p], fontweight='bold')
    axes[row, col].grid('silver', linestyle='-', linewidth=0.4)
    # axes[row, col].set_xlim(0, 0.025)
plt.savefig('../figures/fitting/histo.png', bbox_inches='tight')
plt.close()
