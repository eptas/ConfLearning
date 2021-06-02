import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.weightstats as ssw

from pathlib import Path
from ConfLearning.play.test_experimental_data_simple import run_model, modellist, nsubjects, nblocks, nphases, nbandits, ntrials_phase_max

cwd = Path.cwd()
path_data_r = os.path.join(cwd, '../results/')

nparams = np.zeros(len(modellist))
aic, bic = np.zeros(len(modellist)), np.zeros(len(modellist))

n_subjects = nsubjects
negLogLike = np.full((len(modellist), nsubjects, nblocks, nphases, ntrials_phase_max), np.nan, float)

null_likeli, zero_likeli, conf_likeli = np.zeros(n_subjects), np.zeros(n_subjects), np.zeros(n_subjects)


for m, models in enumerate(modellist):
    for n in range(n_subjects):

        fittingData = pd.read_pickle(os.path.join(path_data_r, 'fittingData/fittingDataM' + str(m) + '.pkl'))
        alpha, beta, alpha_c, gamma, alpha_n = fittingData.ALPHA, fittingData.BETA, fittingData.ALPHA_C, fittingData.GAMMA, fittingData.ALPHA_N

        parameter = [[alpha[n], beta[n]], [alpha[n], beta[n], alpha_c[n]], *[[alpha[n], beta[n], alpha_c[n], gamma[n]] for _ in range(4)],
                     *[[alpha[n], beta[n], alpha_c[n], gamma[n], alpha_n[n]] for _ in range(4)], *[[alpha[n], beta[n], alpha_c[n], gamma[n]] for _ in range(2)],
                     *[[alpha[n], beta[n], alpha_c[n], gamma[n], alpha_n[n]] for _ in range(2)]]

        nparams[m] = len(parameter[m])

        negLogLike[m, n, :, :, :] = run_model(parameter[m], models, n, return_cp=False, return_full=False, return_conf_esti=False, return_nll=True, return_bias=False)


for m, model in enumerate(modellist):

    logLike = np.nanmean(negLogLike[m, :, :, 1, :])
    nsamples = np.sum(~np.isnan(negLogLike[m, :, :, 1, :]))     # len(negLogLike[m, :, :, 1, :][~np.isnan(negLogLike[m, :, :, 1, :])])

    aic[m] = 2 * nparams[m] + 2 * logLike
    bic[m] = nparams[m] * np.log(nsamples) + 2 * logLike

    if m == 0:
        for n in range(n_subjects):
            null_likeli[n] = np.nanmean(negLogLike[m, n, :, 1, :])

    if m == 1:
        for n in range(n_subjects):
            zero_likeli[n] = np.nanmean(negLogLike[m, n, :, 1, :])

    if m == 3:
        for n in range(n_subjects):
            conf_likeli[n] = np.nanmean(negLogLike[m, n, :, 1, :])

model_fit = [aic, bic]
param_name = ['alpha', 'beta', 'alpha_c', 'gamma', 'alpha_n']
model_name = ['Rescorla\nStatic', 'Rescorla\nZero', 'Rescorla\nConf', 'Rescorla\nConfGen', 'Rescorla\nConfNofeed', 'Rescorla\nConfNofeedGen',
              'Rescorla\nConfZero', 'Rescorla\nConfZeroGen', 'Rescorla\nConfNofeedZero', 'Rescorla\nConfNofeedZeroGen', 'BayesModel', 'Bayes\nIdealObserver']


for m, model in enumerate(model_fit):

    plt.figure(m, figsize=(20, 10))
    plt.bar(range(len(modellist)), model, width=0.8, color='g')

    for t in range(len(modellist)):
        plt.text(t, 200, str(round(model[t], 2)), color='w', fontsize=10)

    plt.title('model fit - phase 1 only')
    plt.xlabel('model', fontweight='bold')
    plt.xticks(np.arange(len(modellist)), model_name, fontsize=10)
    plt.ylabel('AIC' if (m == 0) else 'BIC', fontweight='bold')
    plt.yticks(np.arange(0, 250, step=50), fontsize=6)
    plt.grid('silver', linestyle='-', linewidth=0.4)
    plt.savefig('AIC_phase1.png' if (m == 0) else 'BIC_phase1.png', bbox_inches='tight')
    plt.close()

testing = ssw.ttost_paired(null_likeli, conf_likeli, low=0.05, upp=0.95)
print('Null_model testing \n p = ' + str(testing[0]))
print('t1, pv1, df1 = ' + str(testing[1]))
print('t2, pv2, df2 = ' + str(testing[2]))

testing = ssw.ttost_paired(zero_likeli, conf_likeli, low=0.05, upp=0.95)
print('Zero_model testing \n p = ' + str(testing[0]))
print('t1, pv1, df1 = ' + str(testing[1]))
print('t2, pv2, df2 = ' + str(testing[2]))
