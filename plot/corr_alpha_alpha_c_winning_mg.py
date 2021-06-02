import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats
import pingouin as pg

# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))
from plot_util import set_fontsize, savefig  # noqa

cwd = Path.cwd()
path_data = os.path.join(cwd, '../results/fittingData')

models = np.arange(1, 10)
n_subjects = 66

winning_model = 4
# winning_model = 5

fittingData0 = pd.read_pickle(os.path.join(path_data, f'fittingDataM0.pkl'))
fittingData = pd.read_pickle(os.path.join(path_data, f'fittingDataM{winning_model}.pkl'))

plt.figure(figsize=(5, 4))

# rho, pval = stats.spearmanr(fittingData.ALPHA_C, fittingData0.ALPHA)
rho, pval, outliers = pg.correlation.shepherd(fittingData.ALPHA_C, fittingData0.ALPHA)
# corr = np.corrcoef(fittingData.ALPHA_C, fittingData.ALPHA)[0][1]
plt.scatter(fittingData.ALPHA_C, fittingData0.ALPHA, s=8, c=(0.5, 0.5, 0.5), marker='o')
plt.xlabel(r'Confidence learning rate $\alpha_c$')
plt.ylabel(r'Value learning rate $\alpha$')
# plt.xticks(np.arange(0, 1.2, step=0.2))
# plt.yticks(np.arange(0, 1.2, step=0.2))
if pval < 0.001:
    plt.text(0.4, 0.5, fr'$r={rho:.2f}\;\;(p<0.001)$', color='k', fontsize=10)
else:
    plt.text(0.4, 0.5, fr'$r={rho:.2f}\;\;(p={pval:.3f})$', color='k', fontsize=10)
# plt.grid('silver', linestyle='-', linewidth=0.4)

# os.makedirs('../figures/param_corr')
savefig('../figures/param_corr/corr_gamma_alpha_winning_mg.png')
plt.close()
