import os
import sys
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))
from plot_util import set_fontsize, savefig  # noqa

path_data = os.path.join(Path.cwd(), '../data/')

data = pd.read_pickle(os.path.join(path_data, 'data.pkl'))
data = data[~data.subject.isin([25, 30])]
subjects = sorted(data.subject.unique())
nsubjects = len(subjects)

ntrials_phase0 = (9, 12, 15, 18)
ntrials_phase1 = (0, 5, 10, 15)
ntrials_phase2 = (9, 12, 15, 18)
nt_phase1_max = np.max(ntrials_phase1)
nt_phase2_max = np.max(ntrials_phase2)
nt_phase0phase1 = 27
nblocks = 11

colors = sns.color_palette()

valvar = 'value_id'
values = sorted(data[~data[valvar].isna()][valvar].unique())
nvalues = len(values)


reload = False
if reload:
    count, consistent = np.zeros(nsubjects), np.zeros(nsubjects)
    count2, consistent2 = np.zeros(nsubjects), np.zeros(nsubjects)
    count3, consistent3 = np.zeros(nsubjects), np.zeros(nsubjects)
    for s in range(nsubjects):
        print(f'Subject {s + 1} / {nsubjects}')
        for b in range(nblocks):
            d = data[(data.subject == s) & (data.block == b) & (data.phase == 1)]
            for p in d.pair.unique():
                if len(d[d.pair == p]) > 1:
                    # print(len(d[d.pair == p]))
                    trials = d[d.pair == p].trial_phase.values
                    for i, t in enumerate(trials[1:]):
                        count[s] += 1
                        consistency = d[(d.pair == p) & (d.trial_phase == t)].choice.values[0] == d[(d.pair == p) & (d.trial_phase == trials[i])].choice.values[0]
                        consistent[s] += consistency
                        data.loc[(data.subject == s) & (data.block == b) & (data.phase == 1) & (data.trial_phase == t), 'repeat_nr'] = i
                        data.loc[(data.subject == s) & (data.block == b) & (data.phase == 1) & (data.trial_phase == t), 'consistent'] = int(consistency)
                        if i == 0:
                            count2[s] += 1
                            consistent2[s] += consistency
                        elif i == 1:
                            count3[s] += 1
                            consistent3[s] += consistency

    pickle.dump((count, consistent, count2, consistent2, count3, consistent3), open('../results/behav/consistency.pkl', 'wb'))
else:
    count, consistent, count2, consistent2, count3, consistent3 = pickle.load(open('../results/behav/consistency.pkl', 'rb'))


count, consistent = count[np.setdiff1d(range(nsubjects), [25, 30])], consistent[np.setdiff1d(range(nsubjects), [25, 30])]
count2, consistent2 = count2[np.setdiff1d(range(nsubjects), [25, 30])], consistent2[np.setdiff1d(range(nsubjects), [25, 30])]
count3, consistent3 = count3[np.setdiff1d(range(nsubjects), [25, 30])], consistent3[np.setdiff1d(range(nsubjects), [25, 30])]



plt.figure(figsize=(7, 3))

ax1 = plt.subplot(121)
plt.hist(consistent2 / count2, bins=np.arange(0, 1.01, 0.02), facecolor=colors[0], alpha=0.7, label=r'1st $\rightarrow$ 2nd choice', zorder=6)
plt.hist(consistent3 / count3, bins=np.arange(0, 1.01, 0.02), facecolor=colors[1], alpha=0.7, label=r'2nd $\rightarrow$ 3rd choice', zorder=5)

plt.xlabel('Consistency')
plt.ylabel('Number of participants')
plt.xlim(0.5, 1)
plt.legend(loc='upper left')
plt.text(-0.3, 0.97, 'A', transform=ax1.transAxes, color=(0, 0, 0), fontsize=20)


ax2 = plt.subplot(122)
ratingdiff21 = data.groupby(['subject', 'value_id']).ratingdiff21.mean().groupby(level='value_id').mean()
ratingdiff21_se = data.groupby(['subject', 'value_id']).ratingdiff21.mean().groupby(level='value_id').sem()
for i, v in enumerate(values):
    plt.bar(i, ratingdiff21[i], yerr=ratingdiff21_se[i], facecolor=colors[i])

plt.plot([-0.75, nvalues-0.25], [0, 0], 'k-', lw=0.5)
plt.xticks(range(nvalues), ['Lowest', '2nd\nlowest', '2nd\nhighest', 'Highest'], fontsize=9)
plt.yticks(np.arange(-0.02, 0.021, 0.01))
plt.xlabel('CS value level')
plt.ylabel('Rating change (post - pre)')
plt.xlim(-0.5, nvalues-0.5)
plt.ylim(-0.02, 0.02)
plt.text(-0.43, 0.97, 'B', transform=ax2.transAxes, color=(0, 0, 0), fontsize=20)

set_fontsize(label=11, tick=10)
plt.tight_layout()
plt.subplots_adjust(wspace=0.5, left=0.11)
savefig(f'../figures/behav/Figure3.tif', format='tif', dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.show()
