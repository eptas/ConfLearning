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

# model_name = ['Rescorla\nStatic', 'Rescorla\nZero', 'Rescorla\nConf', 'Rescorla\nConfGen', 'Rescorla\nConfNofeed', 'Rescorla\nConfNofeedGen',
#               'Rescorla\nConfZero', 'Rescorla\nConfZeroGen', 'Rescorla\nConfNofeedZero', 'Rescorla\nConfNofeedZeroGen']
model_name = ['NoDeval [-]', 'Deval [-]', 'NoDeval [Spec+Dual]', 'NoDeval [Unspec+Dual]',
              'NoDeval [Spec+Mono]', 'NoDeval [Unspec+Mono]', 'Deval [Spec+Dual]', 'Deval [Unspec+Dual]',
              'Deval [Spec+Mono]', 'Deval [unspec+Mono]']
model_name = ['[-]', '[-]$^\mathrm{d}$', '[dual+spec]', '[dual+unspec]',
              '[mono+spec]', '[mono+unspec]', '[dual+spec]$^\mathrm{d}$', '[dual+unspec]$^\mathrm{d}$',
              '[mono+spec]$^\mathrm{d}$', '[mono+unspec]$^\mathrm{d}$']
n_models = len(model_name)
n_subjects = 66

AIC = np.full((n_models, n_subjects), np.nan)
for n in range(n_models):
    fittingData = pd.read_pickle(os.path.join(path_data, 'fittingData/fittingDataM' + str(n) + '_ConfSim.pkl'))
    AIC[n] = fittingData.AIC

AICm = np.mean(AIC, axis=1)
# we compute the within-subject-corrected standard error
AICe = sem(AIC - AIC.mean(axis=0), axis=1)

order = np.argsort(AICm)[::-1]

plt.figure(figsize=(6, 4))
plt.gca().xaxis.grid('silver', linestyle='-', linewidth=0.4, zorder=-5)
plt.barh(range(n_models), AICm[order], xerr=AICe, color=color_aic, zorder=2)

for t in range(n_models):
    plt.text(400.4, t-0.2, model_name[order[t]], color='k', fontsize=10, zorder=3)

plt.xlabel('AIC')
plt.xticks(range(400, 426, 5))
plt.yticks(range(n_models), [])
plt.xlim(400, 425.5)
plt.ylim(-0.6, n_models-0.4)
set_fontsize(xlabel=12, tick=10)
savefig('../figures/fitting/AIC_ConfSim_MG.png')
plt.close()

