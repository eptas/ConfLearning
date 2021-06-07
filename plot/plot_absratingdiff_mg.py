import os
import sys
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.stats import wilcoxon

# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))
from plot_util import set_fontsize, savefig  # noqa

nsubjects = 66

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

plt.xticks([0, 1], ['Pre', 'Post'])
plt.ylabel('Abs. difference between ratings')
plt.xlim([-0.5, 1.5])
plt.ylim(0.25, 0.28)
plt.tight_layout()
savefig('../figures/behav/absratingdiff.png')