import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from ConfLearning.play.test_experimental_data_simple import run_model, modellist, nsubjects, nblocks, nphases, nbandits

model = 3

cwd = Path.cwd()
path_data_r = os.path.join(cwd, '../results/')

fittingData = pd.read_pickle(os.path.join(path_data_r, 'fittingData/fittingDataM' + str(model) + '.pkl'))

alpha, beta, alpha_c, gamma, alpha_n = fittingData.ALPHA, fittingData.BETA, fittingData.ALPHA_C, fittingData.GAMMA, fittingData.ALPHA_N

pe_curve = np.full((nsubjects, 90), np.nan)
exp_curve = np.full((nsubjects, 90), np.nan)

columns = []

for i in range(nsubjects):
    column_names = 's' + str(i)
    columns = np.append(columns, column_names)

peDF = None
expectDF = None


for m, models in enumerate(modellist):
    for n in range(nsubjects):
        if models != modellist[model]:
            continue

        parameter = [[alpha[n], beta[n]], [alpha[n], beta[n], alpha_c[n]], *[[alpha[n], beta[n], alpha_c[n], gamma[n]] for _ in range(4)],
                     *[[alpha[n], beta[n], alpha_c[n], gamma[n], alpha_n[n]] for _ in range(4)], *[[alpha[n], beta[n], alpha_c[n], gamma[n]] for _ in range(2)],
                     *[[alpha[n], beta[n], alpha_c[n], gamma[n], alpha_n[n]] for _ in range(2)]]

        conf_prediction_error, conf_expected_value = run_model(parameter[m], models, n, return_cp=False, return_full=False, return_conf_esti=True)

        pe_curve = pd.DataFrame(data={"s" + str(n): conf_prediction_error[:, 1, :][~np.isnan(conf_prediction_error[:, 1, :])]},
                                      columns=["s" + str(n)])

        exp_curve = pd.DataFrame(data={"s" + str(n): conf_expected_value[:, 1, :][~np.isnan(conf_expected_value[:, 1, :])]},
                                      columns=["s" + str(n)])

        peDF = pd.concat([peDF, pe_curve], axis=1)
        expectDF = pd.concat([expectDF, exp_curve], axis=1)

peDF_mean = np.nanmean(peDF, axis=1)
peDF_std = np.nanstd(peDF, axis=1)

expectDF_mean = np.nanmean(expectDF, axis=1)
expectDF_std = np.nanstd(expectDF, axis=1)

x = range(len(peDF_std))
y1 = peDF_mean + peDF_std
y2 = peDF_mean - peDF_std

plt.figure(figsize=(20, 10))
plt.plot(x, y1, color="orange")
plt.plot(x, y2, color="orange")
plt.fill_between(x, y1, y2, facecolor="orange", alpha=0.2)
plt.plot(range(len(peDF_mean)), peDF_mean, color='r')
plt.title("confidence PE in phase without feedback")
plt.yticks(np.arange(-2, 7.5, step=1), fontsize=6)
plt.grid('silver', linestyle='-', linewidth=0.4)
plt.savefig('Conf_PE_Curve_M' + str(model) + '.png', bbox_inches='tight')
plt.close()

x = range(len(expectDF_std))
y1 = expectDF_mean + expectDF_std
y2 = expectDF_mean - expectDF_std

plt.figure(figsize=(20, 10))
plt.plot(x, y1, color="orange")
plt.plot(x, y2, color="orange")
plt.fill_between(x, y1, y2, facecolor="orange", alpha=0.2)
plt.plot(range(len(expectDF_mean)), expectDF_mean, color='r')
plt.title("expected confidence in phase without feedback")
plt.yticks(np.arange(-2, 7.5, step=1), fontsize=6)
plt.grid('silver', linestyle='-', linewidth=0.4)
plt.savefig('expected_confidence_curve_M' + str(model) + '.png', bbox_inches='tight')
plt.close()
