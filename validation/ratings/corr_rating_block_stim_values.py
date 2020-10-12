import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats

cwd = Path.cwd()

path_data = os.path.join(cwd, '../../data/')
df = pd.read_pickle(os.path.join(path_data, 'data.pkl'))

block_id = ['block 1', 'block 2', 'block 3', 'block 4', 'block 5', 'block 6', 'block 7', 'block 8', 'block 9', 'block 10']

nsubjects = 66
nblocks = 10

fig, axes = plt.subplots(5, 2, figsize=(20, 10))
rows, columns = [*[0 for _ in range(2)], *[1 for _ in range(2)], *[2 for _ in range(2)], *[3 for _ in range(2)], *[4 for _ in range(2)]], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

stim_combi = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
colors = ['r', 'b', 'g', 'y', 'm', 'tan', 'darkorange', 'darkslategrey', 'teal', 'indianred']

for b in range(nblocks):

    row, col = rows[b], columns[b]

    delta_rating, delta_values = np.full((nsubjects, len(stim_combi)), np.nan), np.full((nsubjects, len(stim_combi)), np.nan)

    for stim, combi in enumerate(stim_combi):
        for s in range(nsubjects):

            rating_post = abs(df[(df.subject == s) & (df.block == b) & df.type_rating2 & (df.stimulus_left == combi[0])].rating.mean() -
                              df[(df.subject == s) & (df.block == b) & df.type_rating2 & (df.stimulus_left == combi[1])].rating.mean())

            rating_pre = abs(df[(df.subject == s) & (df.block == b) & df.type_rating1 & (df.stimulus_left == combi[0])].rating.mean() -
                             df[(df.subject == s) & (df.block == b) & df.type_rating1 & (df.stimulus_left == combi[1])].rating.mean())

            delta_rating[s, stim] = abs(rating_post - rating_pre)

            delta_values[s, stim] = abs(df[(df.subject == s) & (df.block == b) & (df.stimulus_left == combi[0])].stimulus_left_value.mean() -
                                        df[(df.subject == s) & (df.block == b) & (df.stimulus_left == combi[1])].stimulus_left_value.mean())

        rho, pval = stats.spearmanr(delta_values[:, stim], delta_rating[:, stim])
        leg_label = str(combi).replace(",", " -") + ', rho = ' + str(round(rho, 2)) + ', p = ' + str(round(pval, 2))

        axes[row, col].scatter(delta_values[:, stim], delta_rating[:, stim], s=8, facecolors='none', edgecolors=colors[stim], marker='o', label=leg_label)
        axes[row, col].set_title(block_id[b])
        axes[max(rows), col].set_xlabel('abs. value difference (p2-p1)')
        axes[row, 0].set_ylabel('value ratings (p2-p1)')
        axes[row, col].set_xticks(np.arange(0.00, 20.00, step=2.50))
        axes[row, col].set_yticks(np.arange(0.00, 0.70, step=0.2))
        axes[row, col].grid('silver', linestyle='-', linewidth=0.4)

# fig.legend(loc="upper right", ncol=3, fontsize=6, title="stim pairs")
fig.savefig('../../figures/validation/ratings/corr_rating_block_stim_values.png', bbox_inches='tight')
plt.close()
