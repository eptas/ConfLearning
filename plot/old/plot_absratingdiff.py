import os
import pickle
import sys
from itertools import combinations
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem, wilcoxon, ttest_rel

# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))

nsubjects = 66
nblocks = 11

from plot_util import set_fontsize, savefig  # noqa

reload = True

if reload:
    path_data = os.path.join(Path.cwd(), '../data/')
    data = pd.read_pickle(os.path.join(path_data, 'data.pkl'))
    stim_combos = list(combinations(range(5), 2))
    count, absratingdiff1, absratingdiff2 = np.zeros(nsubjects), np.zeros(nsubjects), np.zeros(nsubjects)
    for s in range(nsubjects):
        print(f'Subject {s + 1} / {nsubjects}')
        for b in range(nblocks):
            d = data[(data.subject == s) & (data.block == b) & data.b_has_rating1 & data.b_has_rating2]
            if len(d):
                count[s] += 1
                absratingdiff1[s] += np.mean([np.abs(d[(d.stimulus_left == c[0]) & d.type_rating1].rating.values[0] - d[(d.stimulus_left == c[1]) & d.type_rating1].rating.values[0]) for c in stim_combos])
                absratingdiff2[s] += np.mean([np.abs(d[(d.stimulus_left == c[0]) & d.type_rating2].rating.values[0] - d[(d.stimulus_left == c[1]) & d.type_rating2].rating.values[0]) for c in stim_combos])
    pickle.dump((absratingdiff1 / count, absratingdiff2 / count), open('../results/behav/absratingdiff.pkl', 'wb'))
else:
    absratingdiff1, absratingdiff2 = pickle.load(open('../results/behav/absratingdiff.pkl', 'rb'))

absratingdiff1 = absratingdiff1[np.setdiff1d(range(nsubjects), [25, 30])]
absratingdiff2 = absratingdiff2[np.setdiff1d(range(nsubjects), [25, 30])]
absratingdiff1_cor = absratingdiff1 - np.mean([absratingdiff1, absratingdiff2], axis=0)
absratingdiff2_cor = absratingdiff2 - np.mean([absratingdiff1, absratingdiff2], axis=0)

plt.figure(figsize=(3, 2.5))
plt.bar(0, absratingdiff1.mean(), yerr=sem(absratingdiff1_cor), facecolor=(0.5, 0.5, 0.5))
plt.bar(1, absratingdiff2.mean(), yerr=sem(absratingdiff2_cor), facecolor=(0.5, 0.5, 0.5))

stats = wilcoxon(absratingdiff1, absratingdiff2)
print(f'Absolute pairwise rating difference post vs. pre: W={stats.statistic:.1f} (p={stats.pvalue:.5f})')
stats = ttest_rel(absratingdiff1, absratingdiff2)
print(f'Absolute pairwise rating difference post vs. pre: t={stats.statistic:.1f} (p={stats.pvalue:.5f})')

plt.xticks([0, 1], ['Pre-phase-2', 'Post-phase-2'])
plt.ylabel('Abs. difference between ratings')
plt.xlim([-0.5, 1.5])
plt.ylim(0.25, 0.28)
plt.tight_layout()
savefig('../figures/behav/absratingdiff.png')