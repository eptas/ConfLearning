import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from ConfLearning.play.test_experimental_data_simple import run_model, modellist

cwd = Path.cwd()
path_data = os.path.join(cwd, '../results/fittingData')

model_name = ['Rescorla\nConfZero', 'Rescorla\nConfZeroGen', 'Rescorla\nConfBaseZero', 'Rescorla\nConfBaseZeroGen']

models = modellist[6:10]
alpha_c, alpha_n = np.arange(0, 1, 0.02), np.arange(0, 0.01, 0.0005)
nsubjects = 66


def extract_neg_ll():

    negLL = np.full((len(models), nsubjects, len(alpha_c), len(alpha_n)), np.nan)

    for m, model in enumerate(models):

        fittingData = pd.read_pickle(os.path.join(path_data, 'fittingDataM' + str(m + 6) + '.pkl'))
        alpha, beta, gamma = fittingData.ALPHA, fittingData.BETA, fittingData.GAMMA

        for n in range(nsubjects):
            for c, conf in enumerate(alpha_c):
                for no, feed in enumerate(alpha_n):

                    parameter = [*[[alpha[n], beta[n], alpha_c[c], gamma[n], alpha_n[no]] for _ in range(4)]]

                    negLL[m, n, c, no] = run_model(parameter[m], model, n, return_cp=False, return_full=False)

    return np.save('negLL_alpha_c_n', negLL)


if __name__ == '__main__':

    # extract_neg_ll()
    negLL = np.load('negLL_alpha_c_n.npy')

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    rows, columns = [0, 0, 1, 1], [0, 1, 0, 1]

    for m, model in enumerate(models):

        row, col = rows[m], columns[m]

        for no, feed in enumerate(alpha_n):

            mLikeli = np.mean(negLL[m, :, :, no], axis=0)

            axes[row, col].plot(alpha_c, mLikeli, linewidth=0.5)
            axes[row, col].set_title(model_name[m])
            axes[max(rows), col].set_xlabel('alpha_c')
            axes[row, 0].set_ylabel('neg_log_likelihood')
            axes[row, col].set_yticks(np.arange(205, 236, step=5))
            # axes[row, col].text(alpha_c[no], np.mean(mLikeli), str(round(feed, 2)), color='k', fontsize=8)
            axes[row, col].text(0.5, 220, 'alpha_n (min=0.0, max=0.01, step=0.0005)', fontsize=10)
            axes[row, col].grid('silver', linestyle='-', linewidth=0.4)
            fig.savefig('../figures/validation/likelihood_alpha_c_n.png', bbox_inches='tight')
            plt.close()
