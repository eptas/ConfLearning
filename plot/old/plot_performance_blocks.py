import os
import pickle
import sys
from itertools import combinations
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem, wilcoxon, ttest_rel
from ConfLearning.stats.regression import regression
import pandas as pd


path_data = os.path.join(Path.cwd(), '../data/')
data = pd.read_pickle(os.path.join(path_data, 'data.pkl'))

nsubjects = 66
nblocks = 11


m = np.array(data.groupby('block').correct.mean().values, float)
se = np.array(data.groupby('block').correct.sem().values, float)

plt.figure(figsize=(4, 2.5))
plt.errorbar(range(nblocks), m, yerr=se, lw=2, color='k')
plt.xlabel('Block', fontsize=11)
plt.xticks(range(nblocks), range(1, nblocks+1))
plt.ylabel('Proportion correct', fontsize=11)
plt.ylim(0.5, 1)
plt.tight_layout()
plt.savefig("../figures/behav/performance_blocks.png", bbox_inches='tight', pad_inches=0, dpi=300)

df = pd.DataFrame(index=range(nsubjects * nblocks))
df['subject'] = np.repeat(range(nsubjects), nblocks)
df['block'] = np.tile(range(nblocks), nsubjects)
df['performance'] = np.array(data.groupby(['subject', 'block']).correct.mean().values, float)

regression(df, 'performance ~ block', model_blocks=False, type='ols')