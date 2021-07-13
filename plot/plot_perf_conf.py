import os
import sys
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sns

# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))
from plot_util import set_fontsize, savefig  # noqa

path_data = os.path.join(Path.cwd(), '../data/')

data = pd.read_pickle(os.path.join(path_data, 'data.pkl'))

ntrials_phase0 = (9, 12, 15, 18)
ntrials_phase1 = (0, 5, 10, 15)
ntrials_phase2 = (9, 12, 15, 18)
nt_phase1_max = np.max(ntrials_phase1)
nt_phase2_max = np.max(ntrials_phase2)
nt_phase0phase1 = 27

colors = sns.color_palette()

data = data[data.type_choice & ~data.equal_value_pair & ~data.subject.isin([25, 30])]

# exclude = np.where(data.groupby('subject').correct.mean().values < 0.6)[0]
# data = data[~data.subject.isin(exclude)]

window = 4

plt.figure(figsize=(9, 4))

for sp in range(2):

    plt.subplot(1, 2, sp + 1)

    var = ['correct', 'confidence'][sp]

    for i, nt in enumerate(ntrials_phase0):
        d0 = data[(data.b_ntrials_pre == nt) & (data.phase == 0)].groupby(['subject', 'trial_phase_rev'])[var].mean()
        m = d0.mean(level='trial_phase_rev').rolling(window=window, center=True).mean().values.astype(float)
        se = d0.sem(level='trial_phase_rev').rolling(window=window, center=True).mean().values.astype(float)
        plt.plot(np.arange(-nt+1.5, 1.5), m, lw=2, color=colors[i], alpha=0.6)
        plt.fill_between(np.arange(-nt+1.5, 1.5), m-se, m+se, lw=0, color=colors[i], alpha=0.4)

    plt.axvspan(1, nt_phase1_max, facecolor='0.9', alpha=0.5, zorder=-11)
    # plt.axhspan(0, 0.5, facecolor='0.85', alpha=0.5)
    for i, nt in enumerate(ntrials_phase0):
        d1 = data[(data.b_ntrials_pre == nt) & (data.phase == 1)].groupby(['subject', 'trial_phase'])[var].mean()
        m = d1.mean(level='trial_phase').rolling(window=window, center=True).mean().values.astype(float)
        se = d1.sem(level='trial_phase').rolling(window=window, center=True).mean().values.astype(float)
        plt.plot(np.arange(0.5, nt_phase1_max+0.5), m, lw=2, color=colors[i], alpha=0.6)
        plt.fill_between(np.arange(0.5, nt_phase1_max+0.5), m-se, m+se, lw=0, color=colors[i], alpha=0.4)
    d1 = data[(data.phase == 1)].groupby(['subject', 'trial_phase'])[var].mean()
    m = d1.mean(level='trial_phase').rolling(window=window, center=True).mean().values.astype(float)
    plt.plot(np.arange(0.5, nt_phase1_max+0.5), m, lw=3, color='k', alpha=0.6)

    for i, nt in enumerate(ntrials_phase0):
        d2 = data[(data.b_ntrials_pre == nt) & (data.phase == 2)].groupby(['subject', 'trial_phase'])[var].mean()
        m = d2.mean(level='trial_phase').rolling(window=window, center=True).mean().values.astype(float)
        se = d2.sem(level='trial_phase').rolling(window=window, center=True).mean().values.astype(float)
        plt.plot(np.arange(nt_phase1_max-0.5, nt_phase1_max+nt_phase0phase1-nt-1.5), m, lw=2, color=colors[i], alpha=0.6)
        plt.fill_between(np.arange(nt_phase1_max-0.5, nt_phase1_max+nt_phase0phase1-nt-1.5), m-se, m+se, lw=0, color=colors[i], alpha=0.4)
    if sp == 0:  # performance
        plt.plot([-20, 35], [0.5, 0.5], 'k-', lw=0.5, zorder=-10)
        y_text = 0.43
        plt.yticks(np.arange(0.5, 1, 0.1))
        plt.ylim(0.42, 0.865)
        plt.ylabel('Proportion correct')
    else:  # confidence
        plt.ylim(0, 7)
        y_text = 0.15
        plt.ylabel('Confidence')
    plt.xlim(-20, 35)
    plt.xticks(np.arange(-20, 40, 5))
    plt.text(-10, y_text, 'Phase 1', ha='center', fontsize=11)
    plt.text(8, y_text, 'Phase 2', ha='center', fontsize=11)
    plt.text(25, y_text, 'Phase 3', ha='center', fontsize=11)
    plt.xlabel('Trial')

set_fontsize(label=11, tick=9)

savefig('../figures/behav/perf_conf_over_trials.png')
plt.show()