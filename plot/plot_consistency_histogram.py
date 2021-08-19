import os
import sys
import pandas as pd
from pathlib import Path
import pickle
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sns

# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))
from plot_util import set_fontsize, savefig  # noqa

path_data = os.path.join(Path.cwd(), '../data/')
nsubjects = 66

data = pd.read_pickle(os.path.join(path_data, 'data.pkl'))
data = data[data.type_choice & ~data.equal_value_pair & ~data.subject.isin([25, 30])]
count, consistent, count2, consistent2, count3, consistent3 = pickle.load(open('../results/behav/consistency.pkl', 'rb'))

count, consistent = count[np.setdiff1d(range(nsubjects), [25, 30])], consistent[np.setdiff1d(range(nsubjects), [25, 30])]
count2, consistent2 = count2[np.setdiff1d(range(nsubjects), [25, 30])], consistent2[np.setdiff1d(range(nsubjects), [25, 30])]
count3, consistent3 = count3[np.setdiff1d(range(nsubjects), [25, 30])], consistent3[np.setdiff1d(range(nsubjects), [25, 30])]

colors = sns.color_palette()


plt.figure(figsize=(3.5, 3))
plt.hist(consistent2 / count2, bins=np.arange(0, 1.01, 0.02), facecolor=colors[0], alpha=0.7, label=r'1st $\rightarrow$ 2nd choice', zorder=6)
plt.hist(consistent3 / count3, bins=np.arange(0, 1.01, 0.02), facecolor=colors[1], alpha=0.7, label=r'2nd $\rightarrow$ 3rd choice', zorder=5)

set_fontsize(label=11, tick=10)
plt.xlabel('Consistency')
plt.ylabel('Number of participants')
plt.xlim(0.5, 1)
plt.legend(loc='upper left')

plt.tight_layout()
#
savefig('../figures/behav/consistency_histogram.png')
plt.show()