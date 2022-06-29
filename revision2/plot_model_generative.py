import os
import sys
import warnings
import pickle

import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import sem, linregress
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))
from ConfLearning.plot.plot_util import set_fontsize, savefig  # noqa
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

root = '/home/matteo/Dropbox/python/confidence/ConfLearning/'
HOME = os.path.expanduser("~")

reload = False

if reload:
    files = sorted(os.listdir(os.path.join(root, 'data/sim')))
    nbulk = len(files)
    dfs = [None] * nbulk
    for i, file in enumerate(files):
        dfs[i] = pd.read_pickle(os.path.join(root, 'data/sim', file))
    d = pd.concat(dfs).reset_index(drop=True)
    d.to_pickle(os.path.join(root, 'data/sim_all.pkl.gz'))
else:
    d = pd.read_pickle(os.path.join(root, 'data/sim_all.pkl.gz'))

model_names = ['ConfBaseGen', 'ConfBase', 'ChoiceMono', 'Perseveration']
model_map = dict(
    ConfBaseGen='ConfUnpec',
    ConfBase='ConfSpec',
    ChoiceMono='Choice',
    Perseveration=' Persev.'
)

colors = colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


d = d[(d.model.isin(['Perseveration']) | (d.gamma > 0)) & (d.alpha_c_id != 3)]
# d = d[(d.model == model)]
# c = c[(c.gamma_id > 0)]

fig = plt.figure(figsize=(8, 6.5))
axes = [None] * 32
axc = 0
for m, model in enumerate(model_names):
    if model in ('ChoiceMono', 'Perseveration'):
        alpha_cs = [0]
    else:
        alpha_cs = d[d.model == model].alpha_c.unique()
    gammas = d[d.model == model].gamma.unique()
    ngamma = len(gammas)
    for a, alpha_c in enumerate(alpha_cs):
        df = d[(d.model == model) & (d.alpha_c_id == a)]
        ax = plt.subplot(8, 4, axc + 1)
        axes[axc] = ax
        if (a == 0) & (m == 0):
            plt.title('Performance effect')
        plt.plot([-0.75, ngamma-0.25], [0, 0], 'k-', lw=0.5)
        plt.bar(range(len(gammas)), df.groupby('gamma_id').perf_slope.mean(), yerr=df.groupby('gamma_id').perf_slope.sem(), fc=colors[m])
        plt.xlim(-0.6, ngamma-0.4)
        ylim = (-0.00749, 0.0025)
        plt.ylim(ylim)
        plt.yticks([-0.005, -0.0025, 0, 0.0025], ['-.005', '-.0025', '0', '.0025'])
        # plt.ylim(ylim)
        if (a == len(alpha_cs) - 1) & (m >= 1):
            plt.xticks(range(len(gammas)), [(f'{g:.1g}' if np.abs(g) < 1 else f'{g:.2g}') if m in (2, 3) else (f'{np.log(g):.1g}' if np.log(g) < 1 else f'{np.log(g):.2g}') for g in gammas])
            plt.xlabel({2: '$\lambda$', 3: '$\eta$'}[m] if m in (2, 3) else '$\log \gamma$', labelpad=0)
        else:
            plt.xticks([])
        plt.tick_params(axis='x', which='major', pad=1.5)
        plt.tick_params(axis='y', which='major', pad=0.04)
        if (a == 1) or m in (2, 3):
            plt.text(-0.39, 0.5, model_map[model], transform=ax.transAxes, rotation=90, va='center', fontsize=12, color=colors[m], fontweight='bold')

        ax = plt.subplot(8, 4, axc + 2)
        axes[axc+1] = ax
        if (a == 0) & (m == 0):
            plt.title('Confidence effect')
        plt.plot([-0.75, ngamma-0.25], [0, 0], 'k-', lw=0.5)
        plt.bar(range(len(gammas)), df.groupby('gamma_id').conf_slope.mean(), yerr=df.groupby('gamma_id').conf_slope.sem(), fc=colors[m])
        plt.xlim(-0.6, ngamma-0.4)
        plt.ylim(-0.001, 0.022)
        plt.yticks([0, 0.01, 0.02], ['0', '.01', '.02'])
        if (a == len(alpha_cs) - 1) & (m >= 1):
            plt.xticks(range(len(gammas)), [(f'{g:.1g}' if np.abs(g) < 1 else f'{g:.2g}') if m in (2, 3) else (f'{np.log(g):.1g}' if np.log(g) < 1 else f'{np.log(g):.2g}') for g in gammas])
            plt.xlabel({2: '$\lambda$', 3: '$\eta$'}[m] if m in (2, 3) else '$\log \gamma$', labelpad=0)
        else:
            plt.xticks([])
        plt.tick_params(axis='x', which='major', pad=1.5)
        plt.tick_params(axis='y', which='major', pad=0.04)

        ax = plt.subplot(8, 4, axc + 3)
        axes[axc+2] = ax
        if (a == 0) & (m == 0):
            plt.title('Confidence x value effect')
        plt.plot([-0.75, ngamma-0.25], [0, 0], 'k-', lw=0.5)
        plt.bar(range(len(gammas)), df.groupby('gamma_id').conf_value_slope_full.mean(), yerr=df.groupby('gamma_id').conf_value_slope_full.sem(), fc=colors[m])
        # plt.bar(range(len(gammas)), df.groupby('gamma_id').conf_value_slope.mean(), yerr=df.groupby('gamma_id').conf_value_slope.sem(), fc=colors[m])
        plt.xlim(-0.6, ngamma-0.4)
        plt.ylim(-0.0036, 0.002)
        plt.yticks([-0.002, 0, 0.002], ['-.002', '0', '.002'])
        if (a == len(alpha_cs) - 1) & (m >= 1):
            plt.xticks(range(len(gammas)), [(f'{g:.1g}' if np.abs(g) < 1 else f'{g:.2g}') if m in (2, 3) else (f'{np.log(g):.1g}' if np.log(g) < 1 else f'{np.log(g):.2g}') for g in gammas])
            plt.xlabel({2: '$\lambda$', 3: '$\eta$'}[m] if m in (2, 3) else '$\log \gamma$', labelpad=0)
        else:
            plt.xticks([])
        plt.tick_params(axis='x', which='major', pad=1.5)
        plt.tick_params(axis='y', which='major', pad=0.04)

        ax = plt.subplot(8, 4, axc + 4)
        axes[axc+3] = ax
        if (a == 0) & (m == 0):
            plt.title('Consistency effect')
        plt.plot([-0.75, ngamma-0.25], [0, 0], 'k-', lw=0.5)
        plt.bar(np.arange(len(gammas)), df.groupby('gamma_id').consistency_diff.mean(), yerr=df.groupby('gamma_id').consistency_diff.sem(), fc=colors[m])
        plt.xlim(-0.6, ngamma-0.4)
        plt.ylim(-0.01, 0.17)
        plt.yticks([0, 0.05, 0.1, 0.15], ['0', '.05', '.1', '.15'])
        if (a == len(alpha_cs) - 1) & (m >= 1):
            plt.xticks(range(len(gammas)), [(f'{g:.1g}' if np.abs(g) < 1 else f'{g:.2g}') if m in (2, 3) else (f'{np.log(g):.1g}' if np.log(g) < 1 else f'{np.log(g):.2g}') for g in gammas])
            plt.xlabel({2: '$\lambda$', 3: '$\eta$'}[m] if m in (2, 3) else '$\log \gamma$', labelpad=0)
        else:
            plt.xticks([])
        plt.tick_params(axis='x', which='major', pad=1.5)
        plt.tick_params(axis='y', which='major', pad=0.04)
        if m < 2:
            plt.text(1.01, 0.5, rf'$\alpha_c={alpha_c:.2g}$', transform=ax.transAxes, rotation=-90, va='center', fontsize=10)

        axc += 4

set_fontsize(tick=8, label=10, title=11)
plt.subplots_adjust(hspace=0.1, wspace=0.20, left=0.08, right=0.97, top=0.96, bottom=0.137)

for ax in axes[-8:-4]:
    ax.set_position(Bbox(ax.get_position() + np.array([[0, -0.045], [0, -0.045]])))
for ax in axes[-4:]:
    ax.set_position(Bbox(ax.get_position() + np.array([[0, -0.09], [0, -0.09]])))
for ax in axes[1::4]:
    ax.set_position(Bbox(ax.get_position() + np.array([[-0.007, 0], [-0.007, 0]])))

savefig('../figures/model/model_generative.png')
savefig(f'../figures/model/model_generative.tif', format='tif', dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.show()