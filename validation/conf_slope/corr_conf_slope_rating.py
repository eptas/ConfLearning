import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats

cwd = Path.cwd()

path_data = os.path.join(cwd, '../../data/')
df = pd.read_pickle(os.path.join(path_data, 'data.pkl'))

conf_slope = np.load('confSlope.npy')

nsubjects = 66
delta = np.zeros(nsubjects)


for s in range(nsubjects):

    delta[s] = df[(df.subject == s) & df.type_rating2].rating.mean() - df[(df.subject == s) & df.type_rating1].rating.mean()


rho, pval = stats.spearmanr(conf_slope, delta)
plt.scatter(conf_slope, delta, s=8, c='g', marker='o')
plt.title('Spearman correlation: Confidence slope x value rating difference')
plt.xlabel('confidence slope')
plt.ylabel('normalized value ratings (p2-p1)')
plt.xticks(np.arange(-0.4, 0.42, step=0.2))
# plt.yticks(np.arange(-0.12, 0.13, step=0.05))
plt.text(0, 0 if rho >= 0 else 0, 'rho = ' + str(round(rho, 2)) + ', p = ' + str(round(pval, 2)), color='k', fontsize=10)
plt.grid('silver', linestyle='-', linewidth=0.4)
plt.savefig('../../figures/validation/conf_slope/corr_conf_slope_rating.png', bbox_inches='tight')
plt.close()
