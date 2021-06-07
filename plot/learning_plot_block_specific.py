import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from ConfLearning.play.test_experimental_data_simple import run_model, modellist, nsubjects, nblocks, nphases, nbandits

model = 0

cwd = Path.cwd()
path_data_r = os.path.join(cwd, '../results/')

# os.makedirs('../figures/learning')
# os.makedirs('../results/learning')

fittingData = pd.read_pickle(os.path.join(path_data_r, 'fittingData/fittingDataM' + str(model) + '.pkl'))

alpha, beta, gamma, alpha_c, alpha_n = fittingData.ALPHA, fittingData.BETA, fittingData.GAMMA, fittingData.ALPHA_C, fittingData.ALPHA_N

colors = ['r', 'b', 'g', 'y', 'm']

performance_matrix, learning = None, None

learned_value0, learned_value1, learned_value2, learned_value3, learned_value4 = None, None, None, None, None
mean_value0, mean_value1, mean_value2, mean_value3, mean_value4 = None, None, None, None, None

phase_zero0, phase_zero1, phase_zero2, phase_zero3, phase_zero4 = None, None, None, None, None

phase_one0_5, phase_one1_5, phase_one2_5, phase_one3_5, phase_one4_5 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
phase_one0_10, phase_one1_10, phase_one2_10, phase_one3_10, phase_one4_10 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
phase_one0_15, phase_one1_15, phase_one2_15, phase_one3_15, phase_one4_15 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

for m, models in enumerate(modellist):
    for n in range(nsubjects):
        if models != modellist[model]:
            continue

        parameter = [[alpha[n], beta[n]], [alpha[n], beta[n], gamma[n]], *[[alpha[n], beta[n], gamma[n], alpha_c[n]] for _ in range(4)],
                     *[[alpha[n], beta[n], gamma[n], alpha_c[n], alpha_n[n]] for _ in range(4)], *[[alpha[n], beta[n], gamma[n], alpha_c[n]] for _ in range(2)],
                     *[[alpha[n], beta[n], gamma[n], alpha_c[n], alpha_n[n]] for _ in range(2)]]

        new_value_choice, true_value_choice, performance = run_model(parameter[m], models, n, return_cp=False, return_full=True)

        for k in range(nbandits):
            for b in range(nblocks):
                for p in range(nphases):

                    if p == 0:
                        vals = new_value_choice[b, p, :, k]
                        vals = np.hstack((np.full(np.sum(np.isnan(vals)), np.nan), vals[~np.isnan(vals)]))
                    else:
                        vals = new_value_choice[b, p, :, k][~np.isnan(new_value_choice[b, p, :, k])]

                    learned_values = pd.DataFrame(data={"s" + str(n) + "b" + str(b) + "p" + str(p): vals},
                                                  columns=["s" + str(n) + "b" + str(b) + "p" + str(p)])
                    locals()["learned_value" + str(k)] = pd.concat([eval("learned_value" + str(k)), learned_values], axis=1)

        performances = pd.DataFrame(data={"s" + str(n): performance[:, :, :][~np.isnan(performance[:, :, :])]}, columns=["s" + str(n)])
        performance_matrix = pd.concat([performance_matrix, performances], axis=1)

performance_percent = 100 * performance_matrix[performance_matrix == 1].count(axis=1) / performance_matrix.count(axis=1)

for k in range(nbandits):
    for p in range(nphases):

        mean_values = eval("learned_value" + str(k)).filter(regex="p" + str(p)).mean(axis=1)
        locals()["mean_value" + str(k)] = np.append(eval("mean_value" + str(k)), mean_values[~np.isnan(mean_values)])

    phase_0 = eval("learned_value" + str(k)).filter(regex="p" + str(0)).mean(axis=1)
    locals()["phase_zero" + str(k)] = np.append(eval("phase_zero" + str(k)), phase_0[~np.isnan(phase_0)])

    phase_1 = eval("learned_value" + str(k)).filter(regex="p" + str(1))

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

    a = eval("phase_one" + str(k) + "_5").mean(axis=1)
    b = eval("phase_one" + str(k) + "_10").mean(axis=1)
    c = eval("phase_one" + str(k) + "_15").mean(axis=1)
    plt.plot(range(0, 4), a[0:4], color=colors[k], linewidth=0.5)
    plt.plot(range(0, 9), b[0:9], color=colors[k], linewidth=0.5)
    plt.plot(range(0, 14), c[0:14], color=colors[k], linewidth=0.5)

plt.axvline(0, linewidth=0.5, color='k', linestyle='-')
plt.xlabel('trials across blocks', fontweight='bold')
plt.ylabel('learned bandit value ($real_{min}= 12; real_{max}= 40$)', fontweight='bold')
plt.title('learning curve - ' + str(modellist[model])[38:-2] + 'Static')
plt.text(-10, 25, 'phase_zero', color='k', fontsize=8)
plt.text(10, 25, 'phase_one', color='k', fontsize=8)
plt.xticks(np.arange(-20, 20, step=5), fontsize=6)
plt.yticks(np.arange(0, 51, step=5), fontsize=6)
plt.grid('silver', linestyle='-', linewidth=0.4)
plt.savefig('../figures/learning/M' + str(model) + '_learning_phase_per_block.png', bbox_inches='tight')
plt.close()
