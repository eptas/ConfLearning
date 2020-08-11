import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from ConfLearning.play.test_experimental_data_simple import run_model, modellist, nsubjects, nblocks, nphases, nbandits

model = 11

cwd = Path.cwd()
path_data = os.path.join(cwd, '../results/')

# os.makedirs('../figures/learning')
# os.makedirs('../results/learning')

fittingData = pd.read_pickle(os.path.join(path_data, 'fittingData/fittingDataM' + str(model) + '.pkl'))

alpha, beta, alpha_c, gamma, alpha_n = fittingData.ALPHA, fittingData.BETA, fittingData.ALPHA_C, fittingData.GAMMA, fittingData.ALPHA_N

colors = ['r', 'b', 'g', 'y', 'm']

performance_matrix, learning = None, None

learned_value0, learned_value1, learned_value2, learned_value3, learned_value4 = None, None, None, None, None
mean_value0, mean_value1, mean_value2, mean_value3, mean_value4 = None, None, None, None, None

phase_zero0, phase_zero1, phase_zero2, phase_zero3, phase_zero4 = None, None, None, None, None
phase_one0, phase_one1, phase_one2, phase_one3, phase_one4 = None, None, None, None, None

for m, models in enumerate(modellist):
    for n in range(nsubjects):
        if models != modellist[model]:
            continue

        parameter = [[alpha[n], beta[n]], [alpha[n], beta[n], alpha_c[n]], *[[alpha[n], beta[n], alpha_c[n], gamma[n]] for _ in range(4)],
                     *[[alpha[n], beta[n], alpha_c[n], gamma[n], alpha_n[n]] for _ in range(4)], *[[alpha[n], beta[n], alpha_c[n], gamma[n]] for _ in range(2)]]

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
    # for b in range(nblocks):
    #     for s in range(nsubjects):
    #         vals = eval("learned_value" + str(k)).filter(regex=f's{s}b{b}p0')[f's{s}b{b}p0'].values
    #         learned_values_shifted = pd.DataFrame(np.hstack((np.full(np.sum(np.isnan(vals)), np.nan), vals[~np.isnan(vals)])), columns=[f's{s}b{b}p0'])
    #         locals()["learned_value" + str(k)] = pd.concat([eval("learned_value" + str(k)), learned_values_shifted], axis=1)
    for p in range(nphases):

        mean_values = eval("learned_value" + str(k)).filter(regex="p" + str(p)).mean(axis=1)
        locals()["mean_value" + str(k)] = np.append(eval("mean_value" + str(k)), mean_values[~np.isnan(mean_values)])

    phase_0 = eval("learned_value" + str(k)).filter(regex="p" + str(0)).mean(axis=1)
    locals()["phase_zero" + str(k)] = np.append(eval("phase_zero" + str(k)), phase_0[~np.isnan(phase_0)])

    phase_1 = eval("learned_value" + str(k)).filter(regex="p" + str(1)).mean(axis=1)
    locals()["phase_one" + str(k)] = np.append(eval("phase_one" + str(k)), phase_1[~np.isnan(phase_1)])

plt.figure(0)

for k in range(nbandits):

    x, y = range(len(eval("mean_value" + str(k))) - 1), eval("mean_value" + str(k))[1:len(eval("mean_value" + str(k)))]
    plt.plot(x, y, color=colors[k], linewidth=0.5)

plt.xlabel('trials across blocks', fontweight='bold')
plt.ylabel('learned bandit value ($real_{min}= 12; real_{max}= 40$)', fontweight='bold')
plt.title('learning curve - ' + str(modellist[model])[25:-2])
plt.yticks(np.arange(0, 31, step=5), fontsize=6)
plt.xticks(np.arange(0, len(mean_value0), step=10), fontsize=6)
plt.grid('silver', linestyle='-', linewidth=0.4)
plt.savefig('../figures/learning/M' + str(model) + '_learning.png', bbox_inches='tight')
plt.close()

plt.figure(1)

for k in range(nbandits):

    x, y = range(2 - len(eval("phase_zero" + str(k))), 1), eval("phase_zero" + str(k))[1:len(eval("phase_zero" + str(k)))]
    plt.plot(x, y, color=colors[k], linewidth=0.5)

    w, z = range(len(eval("phase_one" + str(k))) - 1), eval("phase_one" + str(k))[1:len(eval("phase_one" + str(k)))]
    plt.plot(w, z, color=colors[k], linewidth=0.5)

plt.axvline(0, linewidth=0.5, color='k', linestyle='-')
plt.xlabel('trials across blocks', fontweight='bold')
plt.ylabel('learned bandit value ($real_{min}= 12; real_{max}= 40$)', fontweight='bold')
plt.title('learning curve - ' + str(modellist[model])[25:-2])
plt.text(-10, 25, 'phase_zero', color='k', fontsize=8)
plt.text(10, 25, 'phase_one', color='k', fontsize=8)
plt.yticks(np.arange(0, 31, step=5), fontsize=6)
plt.xticks(np.arange(-20, 20, step=5), fontsize=6)
plt.grid('silver', linestyle='-', linewidth=0.4)
plt.savefig('../figures/learning/M' + str(model) + '_learning_phase.png', bbox_inches='tight')
plt.close()

for k in range(nbandits):
    learning = pd.concat([learning, eval("learned_value" + str(k))])
    learning.to_pickle("../results/learning/learningM" + str(model) + ".pkl", protocol=4)
