import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from ConfLearning.play.test_experimental_data_simple import run_model, modellist

cwd = Path.cwd()
path_data = os.path.join(cwd, '../../results/fittingData')
# os.makedirs('../../figures/validation')

model_name = ['RescorlaConf', 'RescorlaConfGen', 'RescorlaConfBase', 'RescorlaConfBaseGen',
              'RescorlaConfZero', 'RescorlaConfZeroGen', 'RescorlaConfBaseZero', 'RescorlaConfBaseZeroGen']

models = modellist[2:10]
alpha_c = np.arange(0, 10, 0.2)
nsubjects = 66

negLL = np.full((len(models), nsubjects, len(alpha_c)), np.nan)

for m, model in enumerate(models):

    fittingData = pd.read_pickle(os.path.join(path_data, 'fittingDataM' + str(m + 2) + '.pkl'))
    alpha, beta, gamma, alpha_n = fittingData.ALPHA, fittingData.BETA, fittingData.GAMMA, fittingData.ALPHA_N

    for n in range(nsubjects):
        for c, conf in enumerate(alpha_c):

            parameter = [*[[alpha[n], beta[n], alpha_c[c], gamma[n]] for _ in range(4)],
                         *[[alpha[n], beta[n], alpha_c[c], gamma[n], alpha_n[n]] for _ in range(4)]]

            negLL[m, n, c] = run_model(parameter[m], model, n, return_cp=False, return_full=False)


fig, axes = plt.subplots(4, 2, figsize=(20, 10))
rows, columns = [*[0 for _ in range(2)], *[1 for _ in range(2)], *[2 for _ in range(2)], *[3 for _ in range(2)]], [0, 1, 0, 1, 0, 1, 0, 1]

for m, model in enumerate(models):

    mLikeli = np.zeros(50)

    for c, conf in enumerate(alpha_c):

        mLikeli[c] = np.mean(negLL[m, :, c])

    row, col = rows[m], columns[m]
    axes[row, col].plot(alpha_c, mLikeli, linewidth=0.5)

    # axes[row, col].set_title(model_name[m])
    axes[max(rows), col].set_xlabel('alpha_c')
    axes[row, col].set_ylabel('neg_log_likelihood')
    # axes[row, col].set_yticks(np.arange(204, 240, step=4))
    # axes[row, col].set_yticks(np.arange(192, 216, step=4))
    axes[row, col].grid('silver', linestyle='-', linewidth=0.4)
    fig.savefig('../../figures/validation/likelihood/likelihood_alpha_c.png', bbox_inches='tight')
    plt.close()
