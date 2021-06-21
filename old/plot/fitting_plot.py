import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

cwd = Path.cwd()
path_data = os.path.join(cwd, '../results/')
# os.makedirs('../figures/fitting')

n_models = 12
n_subjects = 66

AIC, BIC, alpha, beta, gamma, alpha_c, alpha_n = None, None, None, None, None, None, None

for n in range(n_models):

    fittingData = pd.read_pickle(os.path.join(path_data, 'fittingData/fittingDataM' + str(n) + '.pkl'))

    AIC = np.append(AIC, np.mean(fittingData.AIC))
    BIC = np.append(BIC, np.mean(fittingData.BIC))

    locals()["alpha_M" + str(n)] = fittingData.ALPHA
    alpha = np.append(alpha, np.median(fittingData.ALPHA))

    locals()["beta_M" + str(n)] = fittingData.BETA
    beta = np.append(beta, np.median(fittingData.BETA))

    locals()["gamma_M" + str(n)] = fittingData.GAMMA
    gamma= np.append(gamma, np.median(fittingData.GAMMA))

    locals()["alpha_c_M" + str(n)] = fittingData.ALPHA_C
    alpha_c = np.append(alpha_c, np.median(fittingData.ALPHA_C))

    locals()["alpha_n_M" + str(n)] = fittingData.ALPHA_N
    alpha_n = np.append(alpha_n, np.median(fittingData.ALPHA_N))


model_fit = [AIC, BIC]
param_fit = [alpha, beta, gamma, alpha_c, alpha_n]
param_name = ['alpha', 'beta', 'gamma', 'alpha_c', 'alpha_n']
model_name = ['Rescorla\nStatic', 'Rescorla\nZero', 'Rescorla\nConf', 'Rescorla\nConfGen', 'Rescorla\nConfNofeed', 'Rescorla\nConfNofeedGen',
              'Rescorla\nConfZero', 'Rescorla\nConfZeroGen', 'Rescorla\nConfNofeedZero', 'Rescorla\nConfNofeedZeroGen', 'BayesModel', 'Bayes\nIdealObserver']

for m, model in enumerate(model_fit):

    plt.figure(m, figsize=(20, 10))
    plt.bar(range(n_models), model[1:len(model)], width=0.8, color='g')

    for t in range(n_models):
        plt.text(t, 200, str(round(model[1:len(model)][t], 2)), color='w', fontsize=10)

    plt.title('model fit')
    plt.xlabel('model', fontweight='bold')
    plt.xticks(np.arange(n_models), model_name, fontsize=10)
    plt.ylabel('AIC' if (m == 0) else 'BIC', fontweight='bold')
    plt.yticks(np.arange(0, 450, step=50), fontsize=6)
    plt.grid('silver', linestyle='-', linewidth=0.4)
    plt.savefig('../figures/fitting/AIC.png' if (m == 0) else '../figures/fitting/BIC.png', bbox_inches='tight')
    plt.close()

for p, para in enumerate(param_fit):

    plt.figure(p, figsize=(20, 10))
    plt.bar(range(n_models), para[1:len(para)], width=0.8, color='g')

    for t in range(n_models):
        plt.text(t, 0.01, str(round(para[1:len(para)][t], 2)), color='w', fontsize=10)
        plt.scatter(np.full(len(range(n_subjects)), t), eval(param_name[p] + '_M' + str(t)), s=4, c='y', marker='o', zorder=10)

    plt.xlabel('model', fontweight='bold')
    plt.xticks(np.arange(n_models), model_name, fontsize=6)
    # plt.ylim(bottom=0, top=0.025)

    plt.grid('silver', linestyle='-', linewidth=0.4)
    plt.title(param_name[p])
    plt.ylabel(param_name[p] + ' (median)', fontweight='bold')
    plt.savefig('../figures/fitting/' + param_name[p] + '.png', bbox_inches='tight')
    plt.close()