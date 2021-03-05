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

colors = ['r', 'b', 'g', 'y', 'm']

conf_behav_value0, conf_behav_value1, conf_behav_value2, conf_behav_value3, conf_behav_value4 = None, None, None, None, None
mean_value0, mean_value1, mean_value2, mean_value3, mean_value4 = None, None, None, None, None

phase_zero0, phase_zero1, phase_zero2, phase_zero3, phase_zero4 = None, None, None, None, None

phase_one0_5, phase_one1_5, phase_one2_5, phase_one3_5, phase_one4_5 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
phase_one0_10, phase_one1_10, phase_one2_10, phase_one3_10, phase_one4_10 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
phase_one0_15, phase_one1_15, phase_one2_15, phase_one3_15, phase_one4_15 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


for m, models in enumerate(modellist):
    for n in range(nsubjects):
        if models != modellist[model]:
            continue

        parameter = [[alpha[n], beta[n]], [alpha[n], beta[n], alpha_c[n]], *[[alpha[n], beta[n], alpha_c[n], gamma[n]] for _ in range(4)],
                     *[[alpha[n], beta[n], alpha_c[n], gamma[n], alpha_n[n]] for _ in range(4)], *[[alpha[n], beta[n], alpha_c[n], gamma[n]] for _ in range(2)],
                     *[[alpha[n], beta[n], alpha_c[n], gamma[n], alpha_n[n]] for _ in range(2)]]

        conf_prediction_error, conf_expected_value, conf_behavioural = run_model(parameter[m], models, n, return_cp=False, return_full=False, return_conf_esti=True)

        for k in range(nbandits):
            for b in range(nblocks):
                for p in range(nphases):

                    if p == 0:
                        confBEV = conf_behavioural[b, p, :, k]
                        confBEV = np.hstack((np.full(np.sum(np.isnan(confBEV)), np.nan), confBEV[~np.isnan(confBEV)]))
                    else:
                        confBEV = conf_behavioural[b, p, :, k][~np.isnan(conf_behavioural[b, p, :, k])]

                    conf_behav_values = pd.DataFrame(data={"s" + str(n) + "b" + str(b) + "p" + str(p): confBEV},
                                                  columns=["s" + str(n) + "b" + str(b) + "p" + str(p)])
                    locals()["conf_behav_value" + str(k)] = pd.concat([eval("conf_behav_value" + str(k)), conf_behav_values], axis=1)


for k in range(nbandits):
    for p in range(nphases):

        mean_values = eval("conf_behav_value" + str(k)).filter(regex="p" + str(p)).mean(axis=1)
        locals()["mean_value" + str(k)] = np.append(eval("mean_value" + str(k)), mean_values[~np.isnan(mean_values)])

    phase_0 = abs(eval("conf_behav_value" + str(k)).filter(regex="p" + str(0))).mean(axis=1)
    locals()["phase_zero" + str(k)] = np.append(eval("phase_zero" + str(k)), phase_0[~np.isnan(phase_0)])

    phase_1 = eval("conf_behav_value" + str(k)).filter(regex="p" + str(1))

    for column in phase_1:
        index = np.where(phase_1[column].isnull().values == False)

        if len(index[0]) == 5:
            locals()["phase_one" + str(k) + "_5"] = pd.concat([eval("phase_one" + str(k) + "_5"), phase_1[column]], axis=1)
        elif len(index[0]) == 10:
            locals()["phase_one" + str(k) + "_10"] = pd.concat([eval("phase_one" + str(k) + "_10"), phase_1[column]], axis=1)
        elif len(index[0]) == 15:
            locals()["phase_one" + str(k) + "_15"] = pd.concat([eval("phase_one" + str(k) + "_15"), phase_1[column]], axis=1)
        else:
            continue

plt.figure(0)

for k in range(nbandits):

    x, y = range(2 - len(eval("phase_zero" + str(k))), 1), eval("phase_zero" + str(k))[1:len(eval("phase_zero" + str(k)))]
    plt.plot(x, y, color=colors[k], linewidth=0.5)

    a = abs(eval("phase_one" + str(k) + "_5")).mean(axis=1)
    b = abs(eval("phase_one" + str(k) + "_10")).mean(axis=1)
    c = abs(eval("phase_one" + str(k) + "_15")).mean(axis=1)
    plt.plot(range(0, 4), a[0:4], color=colors[k], linewidth=0.5)
    plt.plot(range(0, 9), b[0:9], color=colors[k], linewidth=0.5)
    plt.plot(range(0, 14), c[0:14], color=colors[k], linewidth=0.5)

plt.axvline(0, linewidth=0.5, color='k', linestyle='-')
plt.xlabel('trials across blocks', fontweight='bold')
plt.ylabel('behavioural confidence per bandit', fontweight='bold')
plt.title('behavioural confidence curve')
plt.text(-10, 1.5, 'phase_zero', color='k', fontsize=8)
plt.text(10, 1.5, 'phase_one', color='k', fontsize=8)
plt.xticks(np.arange(-20, 20, step=5), fontsize=6)
# plt.yticks(np.arange(0, 36, step=5), fontsize=6)
plt.grid('silver', linestyle='-', linewidth=0.4)
plt.savefig('behavioural_conf_per_bandit.png', bbox_inches='tight')
plt.close()
