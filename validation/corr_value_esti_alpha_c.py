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
path_results = os.path.join(cwd, '../results/fittingData')

fittingData = pd.read_pickle(os.path.join(path_results, 'fittingDataM' + str(model) + '.pkl'))

alpha, beta, gamma, alpha_c = fittingData.ALPHA, fittingData.BETA, fittingData.GAMMA, fittingData.ALPHA_C
stim_combi = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]

value_post, value_pre = np.full((len(stim_combi), nsubjects), np.nan), np.full((len(stim_combi), nsubjects), np.nan)

for stim, combi in enumerate(stim_combi):
    for n in range(nsubjects):

        parameter = [alpha[n], beta[n], gamma[n], alpha_c[n]]

        new_values_choice, true_values_choice, performance = run_model(parameter, modellist[model], n, return_cp=False, return_full=True)

        value_post[stim, n] = abs(np.nanmean(new_values_choice[:, 2, 1, combi[0]]) - np.nanmean(new_values_choice[:, 2, 1, combi[1]]))

        value_pre[stim, n] = abs(np.nanmean(new_values_choice[:, 1, 1, combi[0]]) - np.nanmean(new_values_choice[:, 1, 1, combi[1]]))

delta_values = np.mean(value_post, axis=0) - np.mean(value_pre, axis=0)


rho, pval = stats.spearmanr(delta_values, fittingData.ALPHA_C)
plt.scatter(delta_values, fittingData.ALPHA_C, s=8, c='g', marker='o')
plt.title('Spearman correlation: abs. value estimate difference x alpha_c')
plt.xlabel('abs. value estimate difference (p2, t1 - p1, t1)')
plt.ylabel('alpha_c')
plt.grid('silver', linestyle='-', linewidth=0.4)
plt.yticks(np.arange(0, 1.1, step=0.2))
plt.text(5, 0.5 if rho >= 0 else 0.5, 'rho = ' + str(round(rho, 2)) + ', p = ' + str(round(pval, 2)), color='k', fontsize=10)
plt.savefig('../figures/validation/corr_value_esti_alpha_c.png', bbox_inches='tight')
plt.close()

