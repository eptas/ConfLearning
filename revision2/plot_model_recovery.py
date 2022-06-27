import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import os
from matplotlib.colors import LinearSegmentedColormap
import sys
from ConfLearning.plot.plot_util import set_fontsize, savefig  # noqa

HOME = os.path.expanduser("~")
root = os.path.join(HOME, 'Dropbox/python/confidence/ConfLearning/results/')
# path = os.path.join(root, 'p_fit_gen_ab.npy')
path = os.path.join(root, 'p_gen_fit_ab.npy')
p_fit_gen_ab = np.load(path)

alphas = np.linspace(0.1, 1, 5)
betas = np.array([0.1, 0.2, 0.4, 0.8, 1.6])
# models = ['Static', 'Deval', 'Choice', 'ConfSpec', 'ConfUnspec', 'Persev.']
models = ['$Static$', '$Deval$', '$Choice$', '$ConfSpec$', '$ConfUnspec$', '$Persev.$']
nmodels = len(models)

params = [r'$\alpha$', r'$\beta$', r'$\alpha_c$', r'$\gamma$']

cmap = LinearSegmentedColormap.from_list('bg', [(0.85, 0.85, 0.85), (0, 0.8, 0)], N=256)
fig = plt.figure(figsize=(9, 6.5))
fig.suptitle('p(fit|gen)' if 'fit_gen' in path else 'p(gen|fit)', fontsize=14, fontweight='bold')
for a, alpha in enumerate(alphas):
    for b, beta in enumerate(betas):
        ax = plt.subplot(5, 5, a * 5 + b + 1)
        if a == 0:
            plt.title(rf'$\beta = {beta:.1f}$', fontsize=11, pad=0)
        d = p_fit_gen_ab[a, b]
        im = plt.imshow(d, vmin=0, vmax=1, cmap=cmap, aspect=0.8)
        if b == 4:
            plt.text(1.01, 0.5, rf'$\alpha={alpha:.1f}$', transform=ax.transAxes, rotation=-90, va='center', fontsize=11)
        if a == 4:
            # plt.xticks(range(6), models, rotation=40)
            plt.xticks(range(6), [])
        else:
            plt.xticks([])
        if b == 0:
            plt.yticks(range(6), models)
        else:
            plt.yticks([])
        for x in range(nmodels):
            for y in range(nmodels):
                plt.text(y, x, f'{d[x, y]:.1f}' if np.abs(d[x, y])>=0.995 else ('', '-')[int(d[x, y] < 0)] + f'{np.abs(d[x, y]):.2f}'[1:], ha='center', va='center', fontsize=8, color='k', fontweight='bold')
        set_fontsize(tick=9)

plt.text(0.01, 0.5, 'Generative models', transform=fig.transFigure, rotation=90, va='center', fontsize=13, fontweight='bold')
plt.text(0.5, 0.03, 'Fitted models', transform=fig.transFigure, ha='center', fontsize=13, fontweight='bold')

# Add color bar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.91, 0.07, 0.0225, 0.835])
cb = fig.colorbar(im, cax=cbar_ax)
cb.ax.set_title('Probability', pad=7, fontsize=11)
cb.ax.tick_params(labelsize=10)

plt.subplots_adjust(left=0.13, right=0.88, wspace=0.02, hspace=0.01, top=0.91, bottom=0.07)
savefig(f"../figures/model/model_recovery_{'fit_gen' if 'fit_gen' in path else 'gen_fit'}.png")
# savefig(f'../figures/model/FigureSX.tif', format='tif', dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.show()