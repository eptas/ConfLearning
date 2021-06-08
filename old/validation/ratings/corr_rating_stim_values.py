import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats

cwd = Path.cwd()

path_data = os.path.join(cwd, '../../data/')
df = pd.read_pickle(os.path.join(path_data, 'data.pkl'))

nsubjects = 66

stim_combi = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
colors = ['r', 'b', 'g', 'y', 'm', 'tan', 'darkorange', 'darkslategrey', 'teal', 'indianred']

delta_rating, delta_values = np.full((nsubjects, len(stim_combi)), np.nan), np.full((nsubjects, len(stim_combi)), np.nan)

for stim, combi in enumerate(stim_combi):
    for s in range(nsubjects):

        rating_post = abs(df[(df.subject == s) & df.type_rating2 & (df.stimulus_left == combi[0])].rating.mean() -
                          df[(df.subject == s) & df.type_rating2 & (df.stimulus_left == combi[1])].rating.mean())

        rating_pre = abs(df[(df.subject == s) & df.type_rating1 & (df.stimulus_left == combi[0])].rating.mean() -
                         df[(df.subject == s) & df.type_rating1 & (df.stimulus_left == combi[1])].rating.mean())

        delta_rating[s, stim] = abs(rating_post - rating_pre)

        delta_values[s, stim] = abs(df[(df.subject == s) & (df.stimulus_left == combi[0])].stimulus_left_value.mean() -
                                    df[(df.subject == s) & (df.stimulus_left == combi[1])].stimulus_left_value.mean())

    rho, pval = stats.spearmanr(delta_values[:, stim], delta_rating[:, stim])
    leg_label = str(combi).replace(",", " -") + ', rho = ' + str(round(rho, 2)) + ', p = ' + str(round(pval, 2))

    plt.scatter(delta_values[:, stim], delta_rating[:, stim], s=8, facecolors='none', edgecolors=colors[stim], marker='o', label=leg_label)
    plt.title('Spearman correlation: abs. value difference x norm. value ratings')
    plt.xlabel('abs. value difference (p2-p1)')
    plt.ylabel('norm. value ratings (p2-p1)')
    plt.grid('silver', linestyle='-', linewidth=0.4)

plt.legend(loc="upper right", ncol=3, fontsize=6, title="stim pairs")
plt.savefig('../../figures/validation/ratings/corr_rating_stim_values.png', bbox_inches='tight')
plt.close()
