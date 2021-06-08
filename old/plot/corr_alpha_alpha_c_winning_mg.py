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

confsim = False
max_alphac = True

winning_model = 'DualUnspec'
# winning_model = 5

# fittingData0 = pd.read_pickle(os.path.join(path_data, f"fittingDataM0{('', '_ConfSim')[confsim]}.pkl"))
# fittingData0 = pd.read_pickle(os.path.join(path_data, f"fittingDataM0{('', '_cp')[confsim]}.pkl"))
fittingData0 = pd.read_pickle(os.path.join(path_data, f"fittingData_{winning_model}{('', '_cp')[confsim]}{('', '_max_alpha_c')[max_alphac]}.pkl"))
# fittingData0 = pd.read_pickle(os.path.join(path_data, f"fittingDataM{winning_model}{('', '_ConfSim')[confsim]}.pkl"))
# fittingData0 = pd.read_pickle(os.path.join(path_data, f'fittingData_phase0.pkl'))
# fittingData = pd.read_pickle(os.path.join(path_data, f"fittingDataM{winning_model}{('', '_ConfSim')[confsim]}.pkl"))
fittingData = pd.read_pickle(os.path.join(path_data, f"fittingData_{winning_model}{('', '_cp')[confsim]}{('', '_max_alpha_c')[max_alphac]}.pkl"))

fig, ax = plt.subplots(figsize=(5, 4))

# rho, pval = stats.spearmanr(fittingData.GAMMA, fittingData0.ALPHA)
# rho, pval = stats.pearsonr(fittingData.GAMMA, fittingData0.ALPHA)
rho, pval, outliers = pg.correlation.shepherd(fittingData0.ALPHA, fittingData.ALPHA_C)
# corr = np.corrcoef(fittingData.GAMMA, fittingData.ALPHA)[0][1]
plt.scatter(fittingData0.ALPHA, fittingData.ALPHA_C, s=8, c=(0.5, 0.5, 0.5), marker='o')
for i in np.where(outliers)[0]:
    plt.plot(fittingData0.ALPHA[i], fittingData.ALPHA_C[i], 'o', markersize=8, mfc='None', mec='k')
plt.xlabel(r'Value learning rate $\alpha$')
plt.ylabel(r'Confidence learning rate $\gamma$')
# plt.xticks(np.arange(0, 1.2, step=0.2))
# plt.yticks(np.arange(0, 1.2, step=0.2))
rp_str = fr'$r={rho:.2f}\;\;(p<0.001)$' if pval < 0.001 else fr'$r={rho:.2f}\;\;(p={pval:.3f})$'
plt.text(0.95, 0.9, rp_str, color='k', fontsize=10, transform=ax.transAxes, ha='right')
# plt.grid('silver', linestyle='-', linewidth=0.4)
# plt.ylim(0, 3)
# os.makedirs('../figures/param_corr')
# savefig(f"../figures/param_corr/corr_alpha_gamma_winning_mg{('', '_ConfSim')[confsim]}.png")
savefig(f"../figures/param_corr/corr_alpha_gamma_winning_mg{('', '_cp')[confsim]}{('', '_max_alpha_c')[max_alphac]}.png")
plt.close()
