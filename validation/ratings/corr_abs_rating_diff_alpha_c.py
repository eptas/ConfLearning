import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats

cwd = Path.cwd()

model = 3

path_data = os.path.join(cwd, '../../data/')
path_results = os.path.join(cwd, '../../results/fittingData')

df = pd.read_pickle(os.path.join(path_data, 'data.pkl'))
fittingData = pd.read_pickle(os.path.join(path_results, 'fittingDataM' + str(model) + '.pkl'))

nsubjects = 66

stim_combi = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]

delta_rating = np.full(nsubjects, np.nan)

rating_post = np.full((len(stim_combi), nsubjects), np.nan)
rating_pre = np.full((len(stim_combi), nsubjects), np.nan)

for stim, combi in enumerate(stim_combi):
    for s in range(nsubjects):

        rating_post[stim, s] = abs(df[(df.subject == s) & df.type_rating2 & (df.stimulus_left == combi[0])].rating.mean() -
                                   df[(df.subject == s) & df.type_rating2 & (df.stimulus_left == combi[1])].rating.mean())

        rating_pre[stim, s] = abs(df[(df.subject == s) & df.type_rating1 & (df.stimulus_left == combi[0])].rating.mean() -
                                  df[(df.subject == s) & df.type_rating1 & (df.stimulus_left == combi[1])].rating.mean())

delta_rating = np.mean(rating_post, axis=0) - np.mean(rating_pre, axis=0)


rho, pval = stats.spearmanr(fittingData.GAMMA, delta_rating)
plt.scatter(fittingData.GAMMA, delta_rating, s=8, c='g', marker='o')
plt.title('Spearman correlation: abs. rating difference x alpha_c')
plt.xlabel('alpha_c')
plt.ylabel('abs. rating difference (p2-p1)')
plt.xticks(np.arange(0, 1.1, step=0.2))
plt.grid('silver', linestyle='-', linewidth=0.4)
plt.text(0.4, 0 if rho >= 0 else 0, 'rho = ' + str(round(rho, 2)) + ', p = ' + str(round(pval, 2)), color='k', fontsize=10)
plt.savefig('../../figures/validation/ratings/corr_abs_rating_diff_alpha_c.png', bbox_inches='tight')
plt.close()
