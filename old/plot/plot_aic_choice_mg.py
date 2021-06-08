import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem

# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))
from plot_util import set_fontsize, savefig  # noqa

cwd = Path.cwd()
path_data = os.path.join(cwd, '../results/')

color_aic = (0.6, 0.6, 0.6)

# model_names = ['Static', 'Deval', 'Mono', 'Dual', 'MonoDeval', 'DualDeval']
# nparams = [2, 3, 3, 3, 4, 4]
# model_labels = ['[-]', '[-]$^\mathrm{d}$', '[mono]', '[dual]', '[mono]$^\mathrm{d}$', '[dual]$^\mathrm{d}$']
model_names = ['Static', 'Deval', 'Mono', 'MonoDeval']
nparams = [2, 3, 3, 4]
model_labels = ['[-]', '[-]$^\mathrm{d}$', '[mono]', '[mono]$^\mathrm{d}$']

n_models = len(model_labels)
n_subjects = 66
# n_subjects = 48

AIC = np.full((n_models, n_subjects), np.nan)
for n in range(n_models):
    fittingData = pd.read_pickle(os.path.join(path_data, f"fittingData_control/fittingData_{model_names[n]}_choice.pkl"))
    # fittingData = pd.read_pickle(os.path.join(path_data, f"fittingData_control/fittingData_{model_names[n]}_choice_simchoice.pkl"))
    AIC[n] = fittingData.AIC

AICm = np.mean(AIC, axis=1)
# we compute the within-subject-corrected standard error
AICe = sem(AIC - AIC.mean(axis=0), axis=1)

AIC_choice = AIC[model_labels.index('[mono]')]

order = np.argsort(AICm)[::-1]

plt.figure(figsize=(6, 4))
plt.gca().xaxis.grid('silver', linestyle='-', linewidth=0.4, zorder=-5)
plt.barh(range(n_models), AICm[order], xerr=AICe, color=color_aic, zorder=2)

for t in range(n_models):
    plt.text(390.4, t - 0.2, f'{model_labels[order[t]]} ({nparams[order[t]]})', color='k', fontsize=10, zorder=3)
    plt.text(AICm[order[t]]+1, t - 0.2, f'{AICm[order[t]]:.1f}', color='k', fontsize=10, zorder=3)

plt.xlabel('AIC')
plt.xticks(range(390, 431, 5))
plt.yticks(range(n_models), [])
plt.xlim(390, 430)
plt.ylim(-0.6, n_models-0.4)
set_fontsize(xlabel=12, tick=10)
# savefig(f"../figures/fitting/AIC_ConfSim_MG{('', '_ConfSim')[confsim]}.png")
# savefig(f"../figures/fitting/AIC_choice_MG.png")
# plt.title('Choice effect + simulated choice/confidence')
# savefig(f"../figures/fitting/AIC_choice_simchoice_MG.png")
plt.title('Choice effect + behav. choice/confidence')
savefig(f"../figures/fitting/AIC_choice_MG.png")
# plt.close()

