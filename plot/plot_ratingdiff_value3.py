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

valvar = 'stimulus_left'
values = sorted(data[valvar].unique())
nvalues = len(values)

# ratings1 = np.array([data[data.type_rating2 & ~data.b_has_rating1 & (data[valvar] == v)].groupby('subject').rating.mean().values for v in values])
# ratings1_se = np.array([data[data.type_rating2 & ~data.b_has_rating1 & (data[valvar] == v)].groupby('subject').rating.sem().values for v in values])
# ratingsdiff = np.extended((nsubjects, len(values)), np.nan)
# for s, sub in enumerate(subjects):
#     ratingsdiff[s] = [data[data.type_rating2 & data.b_has_rating1 & data.b_has_rating2 & (data[valvar] == v) & (data.subject == sub)].rating.mean() - data[data.type_rating1 & data.b_has_rating2 & (data[valvar] == v) & (data.subject == sub)].rating.mean() for v in values]
#
plt.figure(figsize=(4, 3))
#
# for i, v in enumerate(values):
#     plt.bar(i, np.nanmean(ratingsdiff[:, i]), yerr=nansem(ratingsdiff[:, i]), facecolor=colors[i])

# data['absratingdiff21'] = data['ratingdiff21'].abs()
ratingdiff21 = data.groupby(['subject', 'stimulus_left']).ratingdiff21.mean().mean(level='stimulus_left')
ratingdiff21_se = data.groupby(['subject', 'stimulus_left']).ratingdiff21.mean().sem(level='stimulus_left')
for i, v in enumerate(values):
    plt.bar(i, ratingdiff21[i], yerr=ratingdiff21_se[i], facecolor=colors[i])

plt.plot([-0.75, nvalues-0.25], [0, 0], 'k-', lw=0.5)
# plt.arrow(0, -0.02, nvalues-0.95, 0, clip_on=False, width=0.0005, head_length=0.2, head_width=0.0015, lw=0, color='k', length_includes_head=True)
# plt.text(2, -0.0232, 'Value', fontsize=11, fontweight='bold', ha='center')
# plt.xticks([])
plt.xticks(range(nvalues), [f'{v:.1f}' for v in data.groupby(['stimulus_left']).stimulus_left_value.mean()])
plt.yticks(np.arange(-0.02, 0.021, 0.01))
plt.xlabel('Value')
plt.ylabel('Rating change (post - pre)')
plt.xlim(-0.75, nvalues-0.25)
plt.ylim(-0.02, 0.02)
set_fontsize(label=14, tick=11)
plt.tight_layout()
savefig('../figures/behav/ratingdiff_value3.png')
plt.show()


# ratingsdiff = np.extended((nsubjects, len(values)), np.nan)
# absvaluediff = np.extended((nsubjects, len(values)), np.nan)
# for s, sub in enumerate(subjects):
#     print(f'Subject {s + 1} / {nsubjects}')
#     ratingsdiff[s] = [data[data.type_rating2 & data.b_has_rating1 & data.b_has_rating2 & (data[valvar] == v) & (data.subject == sub)].rating.mean() - data[data.type_rating1 & data.b_has_rating2 & (data[valvar] == v) & (data.subject == s)].rating.mean() for v in values]
#     absvaluediff[s] = [data[data.b_has_rating1 & data.b_has_rating2 & ((data.stimulus_left == v) | (data.stimulus_right == v)) & (data.subject == sub) & (data.phase == 0)].absvaluediff.mean() for v in values]

# for s, sub in enumerate(subjects):
#     for b in range(11):
#         d = data[(data.subject == sub) & (data.block == b)]
#         if len(d[data.type_rating2 & data.b_has_rating1]):
#             for v in d[~d.ratingdiff21.isna()].stimulus_left:
#                 data.loc[(data.subject == sub) & (data.block == b), 'absvaluediff_phase1'] = d[(d.phase == 1) & (d.stimulus_left == v)].absvaluediff.mean()


# for s, sub in enumerate(subjects):
#     print(f'Subject {s + 1} / {nsubjects}')
#     for b in range(11):
#         d = data[(data.subject == sub) & (data.block == b)]
#         if len(d[d.type_rating2 & d.b_has_rating1]):
#             for v in d[~d.ratingdiff21.isna()].stimulus_left:
#                 # data.loc[(data.subject == sub) & (data.block == b) & ~data.ratingdiff21.isna() & (data.stimulus_left == v), 'absvaluediff_phase1'] = d[(d.phase == 1) & ((d.stimulus_left == v) | (d.stimulus_right == v))].absvaluediff.mean()
#                 data.loc[(data.subject == sub) & (data.block == b) & ~data.ratingdiff21.isna() & (data.stimulus_left == v), 'absvaluediff_phase1'] = (d[(d.phase == 1) & ((d.stimulus_left == v) | (d.stimulus_right == v))].stimulus_left_value_id - d[(d.phase == 1) & ((d.stimulus_left == v) | (d.stimulus_right == v))].stimulus_right_value_id).abs().mean()
# data['absratingdiff21'] = data['ratingdiff21'].abs()
# print(data[~data.absvaluediff_phase1.isna()].astype(dict(absvaluediff_phase1='float'))[['absratingdiff21', 'absvaluediff_phase1']].corr(method=lambda x, y: spearmanr(x, y)[0]))