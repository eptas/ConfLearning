import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import os
from matplotlib.colors import LinearSegmentedColormap
import sys

# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))
from ConfLearning.plot.util.plot_util import set_fontsize, savefig  # noqa
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

HOME = os.path.expanduser("~")

data_default = np.load(os.path.join(HOME, 'Dropbox/python/confidence/ConfLearning/results/cm_pearson.npy'))
data = np.load(os.path.join(HOME, 'Dropbox/python/confidence/ConfLearning/results/cm_pearson_robust.npy'))
data[np.isnan(data)] = data_default[np.isnan(data)]
# data = np.load(os.path.join(HOME, 'Dropbox/python/confidence/ConfLearning/results/cm2_pearson.npy'))

alphas = np.linspace(0.1, 1, 5)
betas = np.array([0.1, 0.2, 0.4, 0.8, 1.6])
alpha_cs = np.linspace(0.1, 1, 5)
# gammas = np.exp(np.linspace(np.log(1), np.log(100), 5))
gammas = np.exp(np.linspace(0, 4, 5))

combos = list(product(range(len(alphas)), range(len(betas)), range(len(alpha_cs)), range(len(gammas))))
ncombos = len(combos)
nparams = 4
params = [r'$\alpha$', r'$\beta$', r'$\alpha_c$', r'$\gamma$']

cmap = LinearSegmentedColormap.from_list('bg', [(1, 0, 0), (0.85, 0.85, 0.85), (0, 0.8, 0)], N=256)
fig = plt.figure(figsize=(7, 6))
for g, gamma in enumerate(gammas):
    for b, beta in enumerate(betas):
        ax = plt.subplot(5, 5, g*5 + b + 1)
        if g == 0:
            plt.title(rf'$\beta = {beta:.1f}$', fontsize=11, pad=0)
        d = np.mean(data[np.array([i for i, c in enumerate(combos) if (c[3], c[1]) == (g, b)])], axis=0)
        im = plt.imshow(d, vmin=-1, vmax=1, cmap=cmap, aspect=0.8)
        if b == 4:
            plt.text(1.01, 0.5, rf'$\log\gamma={np.log(gamma):.1f}$', transform=ax.transAxes, rotation=-90, va='center', fontsize=11)
        if g == 4:
            plt.xticks(range(4), params)
        else:
            plt.xticks([])
        if b == 0:
            plt.yticks(range(4), params)
        else:
            plt.yticks([])
        for x in range(nparams):
            for y in range(nparams):
                plt.text(y, x, f'{d[x, y]:.1f}' if np.abs(d[x, y])>=0.995 else ('', '-')[int(d[x, y] < 0)] + f'{np.abs(d[x, y]):.2f}'[1:], ha='center', va='center', fontsize=8, color='k', fontweight='bold')

plt.text(0.005, 0.5, 'Generative parameters', transform=fig.transFigure, rotation=90, va='center', fontsize=13, fontweight='bold')
plt.text(0.47, 0.045, 'Fitted parameters', transform=fig.transFigure, ha='center', fontsize=13, fontweight='bold')

# Add color bar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.89, 0.11, 0.0225, 0.765])
cb = fig.colorbar(im, cax=cbar_ax)
cb.ax.set_title('Correlation', pad=7, fontsize=11)
cb.ax.tick_params(labelsize=10)

plt.subplots_adjust(left=0.075, wspace=0.05, hspace=0.03)
# savefig('../figures/param_recovery.png')
savefig(f'../figures/FigureS7.tif', format='tif', dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.show()