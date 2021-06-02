import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from ConfLearning.play.test_experimental_data_simple import run_model, modellist

cwd = Path.cwd()
path_data = os.path.join(cwd, '../../results/fittingData')

model_name = ['Rescorla\nConf', 'Rescorla\nConfGen', 'Rescorla\nConfBase', 'Rescorla\nConfBaseGen',
              'Rescorla\nConfZero', 'Rescorla\nConfZeroGen', 'Rescorla\nConfBaseZero', 'Rescorla\nConfBaseZeroGen']

models = modellist[2:10]
alpha_c = np.arange(0, 1, 0.02)
nsubjects = 66

negLL = np.full((len(models), nsubjects, len(alpha_c)), np.nan)

for m, model in enumerate(models):

    fittingData = pd.read_pickle(os.path.join(path_data, 'fittingDataM' + str(m + 2) + '.pkl'))
    alpha, beta, gamma, alpha_n = fittingData.ALPHA, fittingData.BETA, fittingData.ALPHA_C, fittingData.ALPHA_N

    for n in range(nsubjects):
        for g, gam in enumerate(alpha_c):

            parameter = [*[[alpha[n], beta[n], gamma[n], alpha_c[g]] for _ in range(4)],
                         *[[alpha[n], beta[n], gamma[n], alpha_c[g], alpha_n[n]] for _ in range(4)]]

            negLL[m, n, g] = run_model(parameter[m], model, n, return_cp=False, return_full=False)


fig, axes = plt.subplots(4, 2, figsize=(20, 10))
rows, columns = [*[0 for _ in range(2)], *[1 for _ in range(2)], *[2 for _ in range(2)], *[3 for _ in range(2)]], [0, 1, 0, 1, 0, 1, 0, 1]

for m, model in enumerate(models):

    mLikeli = np.zeros(50)

    for g, gam in enumerate(alpha_c):

        mLikeli[g] = np.mean(negLL[m, :, g])

    row, col = rows[m], columns[m]
    axes[row, col].plot(alpha_c, mLikeli, linewidth=0.5)

    # axes[row, col].set_title(model_name[m])
    axes[max(rows), col].set_xlabel('alpha_c')
    axes[row, col].set_ylabel('neg_log_likelihood')
    # axes[row, col].set_yticks(np.arange(200, 221, step=5))
    axes[row, col].set_yticks(np.arange(192, 216, step=4))
    axes[row, col].grid('silver', linestyle='-', linewidth=0.4)
    fig.savefig('../../figures/validation/likelihood/likelihood_alpha_c.png', bbox_inches='tight')
    plt.close()
