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

rating1_short = data[data.b_ntrials_pre.isin([9, 12]) & data.b_has_rating1 & data.b_has_rating2].groupby('subject').rating1.mean()
rating2_short = data[data.b_ntrials_pre.isin([9, 12]) & data.b_has_rating1 & data.b_has_rating2].groupby('subject').rating2.mean()
rating1_long = data[data.b_ntrials_pre.isin([15, 18]) & data.b_has_rating1 & data.b_has_rating2].groupby('subject').rating1.mean()
rating2_long = data[data.b_ntrials_pre.isin([15, 18]) & data.b_has_rating1 & data.b_has_rating2].groupby('subject').rating2.mean()

# rating1_short_cor = rating1_short - np.nanmean(np.array([rating1_short.values, rating2_short.values, rating1_long.values, rating2_long.values]), axis=0)
# rating2_short_cor = rating2_short - np.nanmean(np.array([rating1_short.values, rating2_short.values, rating1_long.values, rating2_long.values]), axis=0)
# rating1_long_cor = rating1_long - np.nanmean(np.array([rating1_short.values, rating2_short.values, rating1_long.values, rating2_long.values]), axis=0)
# rating2_long_cor = rating2_long - np.nanmean(np.array([rating1_short.values, rating2_short.values, rating1_long.values, rating2_long.values]), axis=0)
rating1_short_cor = rating1_short - np.nanmean(np.array([rating1_short.values, rating2_short.values]), axis=0)
rating2_short_cor = rating2_short - np.nanmean(np.array([rating1_short.values, rating2_short.values]), axis=0)
rating1_long_cor = rating1_long - np.nanmean(np.array([rating1_long.values, rating2_long.values]), axis=0)
rating2_long_cor = rating2_long - np.nanmean(np.array([rating1_long.values, rating2_long.values]), axis=0)

plt.figure(figsize=(4, 2.5))
plt.plot([-0.5, 3.75], [0, 0], lw=0.5)
plt.bar(0, rating1_short.mean(), yerr=rating1_short_cor.sem(), facecolor=colors[0], label='Short phase 1')
plt.bar(1, rating2_short.mean(), yerr=rating2_short_cor.sem(), facecolor=colors[0])
plt.bar(2.25, rating1_long.mean(), yerr=rating1_long_cor.sem(), facecolor=colors[1], label='Long phase 1')
plt.bar(3.25, rating2_long.mean(), yerr=rating2_long_cor.sem(), facecolor=colors[1])

plt.xticks([0, 1, 2.25, 3.25], ['Pre', 'Post', 'Pre', 'Post'])
plt.ylabel('Rating')
plt.xlim([-0.5, 3.75])
plt.ylim(0.537, 0.58)
plt.legend(loc='upper right', fontsize=9)
plt.tight_layout()
savefig('../figures/behav/ratingdiff_ntrials_phase0.png')