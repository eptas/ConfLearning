import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import sem
from sklearn.metrics import ConfusionMatrixDisplay as CMD
import matplotlib.transforms

# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))
from plot_util import set_fontsize, savefig  # noqa

cwd = Path.cwd()


# BIC
path_data = os.path.join(cwd, '../results/')
color_BIC = (0.6, 0.6, 0.6)
confsim = False
suffix_conf = '_simchoice'
suffix_choice = '_choice_simchoice'
nparams_conf = [2, 3, 4, 4, 3]
nparams_choice = [3]
nparams = nparams_conf + nparams_choice
names_conf = ['Static', 'Deval', 'MonoSpec', 'MonoUnspec', 'Perservation']
names_choice = ['Mono']
names = names_conf + names_choice
labels_conf = ['Static', 'Deval', 'ConfSpec', 'ConfUnspec', 'Perseveration']
labels_choice = ['Choice']
labels = labels_conf + labels_choice
n_models_conf, n_models_choice, n_models = len(labels_conf), len(labels_choice), len(labels)
n_subjects = 66
BIC = np.full((n_models, n_subjects), np.nan)
for n in range(n_models_conf):
    fittingData = pd.read_pickle(os.path.join(path_data, f"fittingData/fittingData_{names_conf[n]}{suffix_conf}.pkl"))
    BIC[n] = fittingData.BIC
for n in range(n_models_choice):
    fittingData = pd.read_pickle(os.path.join(path_data, f"fittingData/fittingData_{names_choice[n]}{suffix_choice}.pkl"))
    BIC[n+n_models_conf] = fittingData.BIC
BIC = BIC[:, np.setdiff1d(range(n_subjects), [25, 30])]
BICm = np.mean(BIC, axis=1)
# we compute the within-subject-corrected standard error
BICe = sem(BIC - BIC.mean(axis=0), axis=1)
order = np.argsort(BICm)[::-1]


# Confusion matrix
path_data = os.path.join(cwd, '../data/')
model_names = ['Static', 'Deval', 'Choice', 'ConfSpec', 'ConfUnspec', 'Perservation']
modellist = ['Static', 'Deval', 'Mono_choice', 'MonoSpec', 'MonoUnspec', 'Perservation']
n_models = len(model_names)
fitlist = ['NEGLL', 'BIC', 'BIC']
data = pd.read_pickle('data/fittingData_' + model_names[(n_models - 1)] + '.pkl')
BIC = data.filter(regex='BIC_').mean(axis=0)
conf_BIC = np.array([BIC[0:6].values, BIC[6:12].values, BIC[12:18].values, BIC[18:24].values, BIC[24:30].values, BIC[30:36].values]).T
con_BIC = CMD(conf_BIC)



plt.figure(figsize=(5.5, 4))
# plt.gcf().set_size_inches((1.15*4.5, 1.15*3.9))

plt.gca().xaxis.grid('silver', linestyle='-', linewidth=0.4, zorder=-5)
plt.barh(range(n_models), BICm[order], xerr=BICe, color=color_BIC, zorder=2)
for t in range(n_models):
    plt.text(400.35, t-0.15, f'{labels[order[t]]}', color='k', fontsize=13.5, zorder=3, style='italic')
    plt.text(BICm[order[t]]+1, t-0.13, f'{BICm[order[t]]:.1f} ({nparams[order[t]]})', color='k', fontsize=12, zorder=3)
plt.xlabel('BIC')
plt.xticks(range(400, 441, 5))
plt.yticks(range(n_models), [])
plt.xlim(400, 444)
plt.ylim(-0.6, n_models-0.4)


set_fontsize(xlabel=14, tick=11)
plt.tight_layout()
savefig('../figures/model/BIC.png')
savefig(f'../figures/behav/Figure4.tif', format='tif', dpi=600, pil_kwargs={"compression": "tiff_lzw"})