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


models = np.arange(1, 10)
n_subjects = 66

colors = sns.color_palette()

winning_model = 'MonoUnspec'

# suffix = ''
suffix = '_simchoice'
# suffix = '_cp_simchoice'

include = np.setdiff1d(range(n_subjects), [25, 30])

path = Path(__file__).parent
fittingData = pd.read_pickle(os.path.join(path, '../../results/fittingData/', f"fittingData_{winning_model}{suffix}.pkl"))
# fittingData_alpha = pd.read_pickle(os.path.join(path_data, f"fittingData_{winning_model}{suffix}.pkl"))
fittingData_alpha = pd.read_pickle(os.path.join(path, '../../results/fittingData/',  f"fittingData_Static_simchoice.pkl"))


alpha = fittingData_alpha.ALPHA[np.setdiff1d(range(n_subjects), [25, 30])]
gamma = fittingData.GAMMA[np.setdiff1d(range(n_subjects), [25, 30])]


def plot_corr_alpha_gamma():
    rho, pval = pearsonr(alpha, gamma)
    stats = linregress(alpha, gamma)
    plt.plot([0, 0.75], stats.intercept + stats.slope*np.array([0, 0.75]), color=(0.3, 0.3, 0.3), lw=1.5)
    plt.scatter(alpha, gamma, s=40, marker='o', color=colors[0], edgecolors='none', clip_on=False)
    plt.xlabel(r'Reward learning rate $\alpha_r$')
    plt.ylabel(r'Confidence transfer $\gamma$')
    rp_str = fr'$r={rho:.2f}\;\;(p<0.001)$' if pval < 0.001 else fr'$r={rho:.2f}\;\;(p={pval:.3f})$'
    plt.text(0.04, 0.85, rp_str, color='k', fontsize=10, transform=plt.gca().transAxes, ha='left')
    plt.xlim(0, 0.75)
    plt.ylim(0, 55.5)





if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(4, 3.5))
    plt.tight_layout()
    set_fontsize(label=12, tick=11)
    savefig(f"../figures/model/corr_alpha_gamma_winning.png")
    plt.show()