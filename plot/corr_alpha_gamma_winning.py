import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import spearmanr
import pingouin as pg
import seaborn as sns
from scipy.stats import linregress, pearsonr

# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))
from plot_util import set_fontsize, savefig  # noqa

cwd = Path.cwd()
path_data = os.path.join(cwd, '../results/fittingData')

models = np.arange(1, 10)
n_subjects = 66

colors = sns.color_palette()

winning_model = 'MonoUnspec'

# suffix = ''
suffix = '_simchoice'
# suffix = '_cp_simchoice'

include = np.setdiff1d(range(n_subjects), [25, 30])


fittingData = pd.read_pickle(os.path.join(path_data, f"fittingData_{winning_model}{suffix}.pkl"))
# fittingData_alpha = pd.read_pickle(os.path.join(path_data, f"fittingData_{winning_model}{suffix}.pkl"))
fittingData_alpha = pd.read_pickle(os.path.join(path_data, f"fittingData_Static_simchoice.pkl"))


alpha = fittingData_alpha.ALPHA[np.setdiff1d(range(n_subjects), [25, 30])]
gamma = fittingData.GAMMA[np.setdiff1d(range(n_subjects), [25, 30])]

fig, ax = plt.subplots(figsize=(4, 3.5))

rho, pval = pearsonr(alpha, gamma)
# rho, pval = pearsonr(fittingData_alpha.ALPHA, fittingData.GAMMA)
stats = linregress(alpha, gamma)
plt.plot([0, 0.75], stats.intercept + stats.slope*np.array([0, 0.75]), color=(0.3, 0.3, 0.3), lw=1.5)
plt.scatter(alpha, gamma, s=40, marker='o', color=colors[0], edgecolors='none', clip_on=False)
# plt.scatter(fittingData_alpha.ALPHA[~outliers], fittingData.GAMMA[~outliers], s=40, marker='o', color=colors[0], edgecolors='none', clip_on=False)
# plt.scatter(fittingData_alpha.ALPHA[outliers], fittingData.GAMMA[outliers], s=40, marker='o', color=colors[0], alpha=1/3, edgecolors='none')
# for i in np.where(outliers)[0]:
#     plt.plot(fittingData_alpha.ALPHA[i], fittingData.GAMMA[i], 'o', markersize=8, mfc='None', mec=(0.55, 0.55, 0.55), alpha=0.5)
plt.xlabel(r'Value learning rate $\alpha$')
plt.ylabel(r'Confidence learning rate $\gamma$')
rp_str = fr'$r={rho:.2f}\;\;(p<0.001)$' if pval < 0.001 else fr'$r={rho:.2f}\;\;(p={pval:.3f})$'
plt.text(0.04, 0.85, rp_str, color='k', fontsize=10, transform=ax.transAxes, ha='left')
# rp_str = fr'$r={r:.2f}\;\;(p<0.001)$' if p < 0.001 else fr'$r={r:.2f}\;\;(p={p:.3f})$'
# plt.text(0.6, 0.83, rp_str, color='k', fontsize=11, transform=ax.transAxes, ha='left', alpha=0.6)
plt.xlim(0, 0.75)
plt.ylim(0, 55.5)
set_fontsize(label=12, tick=11)
plt.tight_layout()
savefig(f"../figures/model/corr_alpha_gamma_winning.png")
# plt.close()
