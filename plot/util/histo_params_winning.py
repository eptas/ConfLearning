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


winning = 'MonoUnspec_simchoice'
n_subjects = 66

colors = sns.color_palette()

path = Path(__file__).parent
fittingData = pd.read_pickle(os.path.join(path, '../../results/fittingData/', f"fittingData_{winning}.pkl"))

alpha = fittingData.ALPHA[np.setdiff1d(range(n_subjects), [25, 30])]
beta = fittingData.BETA[np.setdiff1d(range(n_subjects), [25, 30])]
gamma = np.log(fittingData.GAMMA[np.setdiff1d(range(n_subjects), [25, 30])][fittingData.GAMMA[np.setdiff1d(range(n_subjects), [25, 30])] != 0])
alpha_c = fittingData.ALPHA_C[np.setdiff1d(range(n_subjects), [25, 30])]

param_fit = dict(alpha=alpha, beta=beta, gamma=gamma, alpha_c=alpha_c)
param_name = dict(alpha=r'$\alpha$', beta=r'$\beta$', gamma=r'$\log(\gamma)}$', alpha_c=r'$\alpha_c$')
xlim = dict(alpha=(0, 1), beta=(0, np.round(beta.max())), gamma=(-0.5, np.ceil(gamma.max())), alpha_c=(0, 1))
ylim = dict(alpha=(0, 10), beta=(0, 20), gamma=(0, 12), alpha_c=(0, 32))
model_name = 'Rescorla\nConfGen'



def plot_histo(param):
    plt.hist(param_fit[param], bins=32, color=(0.4, 0.4, 0.4))
    plt.plot([param_fit[param].mean(), param_fit[param].mean()], ylim[param], 'b-', lw=1.5)
    plt.plot([np.median(param_fit[param]), np.median(param_fit[param])], ylim['alpha_c'], 'g--', lw=1.5)
    plt.title(f'Histogram {param_name[param]}')
    plt.yticks(np.arange(0, 42, step=5))
    plt.xlim(xlim[param])
    plt.ylim(ylim[param])
    if param in ('alpha', 'alpha_c'):
        plt.xticks(np.arange(0, 1.1, 0.2))

    if param == 'alpha_c':
        plt.gca().get_children()[0].set_color(np.array([170, 0, 0])/255)
        inset_axes(plt.gca(), width='100%', height='100%', bbox_to_anchor=(.35, .4, .55, .5), bbox_transform=plt.gca().transAxes)
        plt.hist(alpha_c[alpha_c < 0.02], bins=18, color=np.array([170, 0, 0])/255)
        plt.plot([np.median(param_fit[param]), np.median(param_fit[param])], ylim['alpha_c'], 'g--', lw=1.5)
        plt.ylim(0, 15)
        plt.xlim(0, 0.02)

        plt.xticks(np.arange(0, 0.021, 0.02))

if __name__ == '__main__':

    params = ('alpha', 'beta', 'gamma', 'alpha_c')

    fig, ax = plt.subplots(2, 2, figsize=(9, 3))
    for p, param in enumerate(params):
        ax = plt.subplot(1, 4, p + 1)
        plot_histo(param)

    plt.tight_layout()
    savefig('../figures/model/histo.png')
