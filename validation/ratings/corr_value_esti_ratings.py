import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from pathlib import Path
from ConfLearning.play.test_experimental_data_simple import run_model, modellist


model = 3
nsubjects = 66

cwd = Path.cwd()
path_data = os.path.join(cwd, '../../data/')
path_results = os.path.join(cwd, '../../results/fittingData')

df = pd.read_pickle(os.path.join(path_data, 'data.pkl'))
fittingData = pd.read_pickle(os.path.join(path_results, 'fittingDataM' + str(model) + '.pkl'))

alpha, beta, gamma, alpha_c = fittingData.ALPHA, fittingData.BETA, fittingData.GAMMA, fittingData.ALPHA_C
stim_combi = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]

delta_rating = np.full(nsubjects, np.nan)
rating_post, rating_pre = np.full((len(stim_combi), nsubjects), np.nan), np.full((len(stim_combi), nsubjects), np.nan)

value_post, value_pre = np.full((len(stim_combi), nsubjects), np.nan), np.full((len(stim_combi), nsubjects), np.nan)

for stim, combi in enumerate(stim_combi):
    for n in range(nsubjects):

        parameter = [alpha[n], beta[n], gamma[n], alpha_c[n]]

        new_values_choice, true_values_choice, performance = run_model(parameter, modellist[model], n, return_cp=False, return_full=True)

        value_post[stim, n] = abs(np.nanmean(new_values_choice[:, 2, 1, combi[0]]) - np.nanmean(new_values_choice[:, 2, 1, combi[1]]))

        value_pre[stim, n] = abs(np.nanmean(new_values_choice[:, 1, 1, combi[0]]) - np.nanmean(new_values_choice[:, 1, 1, combi[1]]))

        rating_post[stim, n] = abs(df[(df.subject == n) & df.type_rating2 & (df.stimulus_left == combi[0])].rating.mean() -
                                   df[(df.subject == n) & df.type_rating2 & (df.stimulus_left == combi[1])].rating.mean())

        rating_pre[stim, n] = abs(df[(df.subject == n) & df.type_rating1 & (df.stimulus_left == combi[0])].rating.mean() -
                                  df[(df.subject == n) & df.type_rating1 & (df.stimulus_left == combi[1])].rating.mean())

delta_rating = np.mean(rating_post, axis=0) - np.mean(rating_pre, axis=0)
delta_values = np.mean(value_post, axis=0) - np.mean(value_pre, axis=0)


rho, pval = stats.spearmanr(delta_values, delta_rating)
plt.scatter(delta_values, delta_rating, s=8, c='g', marker='o')
plt.title('Spearman correlation: abs. rating difference x abs. value estimate difference')
plt.xlabel('abs. value estimate difference (p2, t1 - p1, t1)')
plt.ylabel('abs. rating difference (p2 - p1)')
plt.grid('silver', linestyle='-', linewidth=0.4)
plt.text(0.4, 0 if rho >= 0 else 0, 'rho = ' + str(round(rho, 2)) + ', p = ' + str(round(pval, 2)), color='k', fontsize=10)
plt.savefig('../../figures/validation/ratings/corr_abs_rating_diff_value_esti.png', bbox_inches='tight')
plt.close()

