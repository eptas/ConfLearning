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

rating1 = data[data.b_has_rating1 & data.b_has_rating2].groupby('subject').rating1.mean()
rating2 = data[data.b_has_rating1 & data.b_has_rating2].groupby('subject').rating2.mean()
rating1_cor = rating1 - np.nanmean(np.array([rating1.values, rating2.values]), axis=0)
rating2_cor = rating2 - np.nanmean(np.array([rating1.values, rating2.values]), axis=0)

plt.figure(figsize=(4, 2.5))
plt.bar(0, rating1.mean(), yerr=rating1_cor.sem(), facecolor=(0.5, 0.5, 0.5))
plt.bar(1, rating2.mean(), yerr=rating2_cor.sem(), facecolor=(0.5, 0.5, 0.5))

plt.xticks([0, 1], ['Pre', 'Post'])
plt.ylabel('Rating')
plt.xlim([-0.5, 1.5])
plt.ylim(0.55, 0.56)
plt.tight_layout()
savefig('../figures/behav/ratingdiff.png')