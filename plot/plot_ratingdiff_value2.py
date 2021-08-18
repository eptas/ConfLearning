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
from ConfLearning.stats.regression import regression
from ConfLearning.stats.util import nansem

path_data = os.path.join(Path.cwd(), '../data/')

data = pd.read_pickle(os.path.join(path_data, 'data.pkl'))
data = data[~data.subject.isin([25, 30])]
subjects = sorted(data.subject.unique())
nsubjects = len (subjects)

ntrials_phase0 = (9, 12, 15, 18)
ntrials_phase1 = (0, 5, 10, 15)
ntrials_phase2 = (9, 12, 15, 18)
nt_phase1_max = np.max(ntrials_phase1)
nt_phase2_max = np.max(ntrials_phase2)
nt_phase0phase1 = 27

colors = sns.color_palette()

valvar = 'value_chosen_rel3'
values = sorted(data[valvar].unique())
nvalues = len(values)

ratings1 = np.array([data[data.type_rating2 & ~data.b_has_rating1 & (data[valvar] == v)].groupby('subject').rating.mean().values for v in values])
ratings1_se = np.array([data[data.type_rating2 & ~data.b_has_rating1 & (data[valvar] == v)].groupby('subject').rating.sem().values for v in values])
ratingsdiff = np.full((nsubjects, len(values)), np.nan)
for s, sub in enumerate(subjects):
    ratingsdiff[s] = [data[data.type_rating2 & data.b_has_rating1 & (data[valvar] == v) & (data.subject == s)].rating.mean() - data[data.type_rating1 & (data[valvar] == v) & (data.subject == s)].rating.mean() for v in values]

plt.figure(figsize=(4, 3))

for i, v in enumerate(values):
    plt.bar(i, np.nanmean(ratingsdiff[:, i]), yerr=nansem(ratingsdiff[:, i]))
plt.plot([0, nvalues], [0, 0], 'k-', lw=0.5)
plt.xticks(range(nvalues), values)

plt.tight_layout()
set_fontsize(label=11, tick=9)
savefig('../figures/behav/ratingdiff_value2.png')
plt.show()