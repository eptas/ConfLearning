import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import sys
import seaborn as sns

# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))
from plot_util import set_fontsize, savefig  # noqa
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from ConfLearning.stats.regression import linear_regression

cwd = Path.cwd()
path_data = os.path.join(cwd, '../results/')

winning = 'MonoUnspec_simchoice'
n_subjects = 66

colors = sns.color_palette()


fittingData = pd.read_pickle(os.path.join(path_data, f'fittingData_control/fittingData_{winning}.pkl'))

alpha = fittingData.ALPHA[np.setdiff1d(range(n_subjects), [25, 30])]
beta = fittingData.BETA[np.setdiff1d(range(n_subjects), [25, 30])]
gamma = np.log(fittingData.GAMMA[np.setdiff1d(range(n_subjects), [25, 30])][fittingData.GAMMA[np.setdiff1d(range(n_subjects), [25, 30])] != 0])
alpha_c = fittingData.ALPHA_C[np.setdiff1d(range(n_subjects), [25, 30])]

param_fit = [alpha, beta, gamma, alpha_c]
param_name = [r'$\alpha$', r'$\beta$', r'$\log(\gamma)}$', r'$\alpha_c$']
xlim = [(0, 1), (0, np.round(beta.max())), (-0.5, np.ceil(gamma.max())), (0, 1)]
ylim = [(0, 10), (0, 20), (0, 12), (0, 32)]
model_name = 'Rescorla\nConfGen'

fig, ax = plt.subplots(2, 2, figsize=(9, 3))

for p, para in enumerate(param_fit):
    ax = plt.subplot(1, 4, p + 1)
    patches = plt.hist(para, bins=32, color=(0.4, 0.4, 0.4))
    plt.title(f'Histogram {param_name[p]}')
    plt.yticks(np.arange(0, 42, step=5))
    plt.xlim(xlim[p])
    plt.ylim(ylim[p])
    # plt.ylim(0, 18)
    if p in (0, 3):
        plt.xticks(np.arange(0, 1.1, 0.2))
    # if p == 3:
    #     plt.ylim(0, 18)
    # if p > 0:
    #     plt.yticks([])

ax.get_children()[0].set_color(colors[0])
axi = inset_axes(ax, width='100%', height='100%', bbox_to_anchor=(.35, .4, .55, .5), bbox_transform=ax.transAxes)
plt.hist(alpha_c[alpha_c < 0.02], bins=18, color=colors[0])
plt.ylim(0, 15)
plt.xlim(0, 0.02)
plt.xticks(np.arange(0, 0.021, 0.02))
# mark_inset(ax, axi, loc1=2, loc2=4, fc="none", ec="0.5")

plt.tight_layout()
savefig('../figures/fitting/histo_mg.png')
