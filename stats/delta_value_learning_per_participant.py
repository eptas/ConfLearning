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

alpha, beta, gamma, alpha_c, alpha_n = fittingData.ALPHA, fittingData.BETA, fittingData.ALPHA_C, fittingData.GAMMA, fittingData.ALPHA_N

colors = ['r', 'b', 'g', 'y', 'm']

learned_value0, learned_value1, learned_value2, learned_value3, learned_value4 = None, None, None, None, None

diff_matrix = np.full((nsubjects, nbandits), np.nan, float)

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


for k in range(nbandits):
    for s in range(nsubjects):

        first_array = eval("learned_value" + str(k)).filter(regex="p1").filter(regex="b0").filter(regex=("s" + str(s) + "b")).values

        if first_array[~np.isnan(first_array)].size > 0:
            initial_val = first_array[~np.isnan(first_array)][0]
        else:
            first_array = eval("learned_value" + str(k)).filter(regex="p1").filter(regex="b1p").filter(regex=("s" + str(s) + "b")).values

            if first_array[~np.isnan(first_array)].size > 0:
                initial_val = first_array[~np.isnan(first_array)][0]
            else:
                first_array = eval("learned_value" + str(k)).filter(regex="p1").filter(regex="b2p").filter(regex=("s" + str(s) + "b")).values
                initial_val = first_array[~np.isnan(first_array)][0]

        last_array = eval("learned_value" + str(k)).filter(regex="p1").filter(regex="b10").filter(regex=("s" + str(s) + "b")).values

        if last_array[~np.isnan(last_array)].size > 0:
            final_val = last_array[~np.isnan(last_array)][-1]
        else:
            last_array = eval("learned_value" + str(k)).filter(regex="p1").filter(regex="b9").filter(regex=("s" + str(s) + "b")).values
            final_val = last_array[~np.isnan(last_array)][-1]

        diff_matrix[s, k] = final_val - initial_val


plt.figure(0)

for k in range(nbandits):

    y = diff_matrix[:, k]
    plt.scatter(range(len(diff_matrix[:, k])), y, color=colors[k])
    # plt.plot(range(len(diff_matrix[:, k])), y, color=colors[k])

plt.xlabel('participants', fontweight='bold')
plt.ylabel('final value - initial value in p1 per bandit', fontweight='bold')
plt.title('learned value slope per bandit - ' + str(modellist[model])[38:-2])
# plt.xticks(np.arange(-20, 20, step=5), fontsize=6)
# plt.yticks(np.arange(0, 36, step=5), fontsize=6)
plt.grid('silver', linestyle='-', linewidth=0.4)
plt.savefig('value_diff_per_participant_bandit_M' + str(model) + '.png', bbox_inches='tight')
plt.close()

plt.figure(1)

means = np.zeros(nbandits)
stds = np.zeros(nbandits)

x_position = np.arange(nbandits)

for k in range(nbandits):

    means[k] = np.nanmean(diff_matrix[:, k])
    stds[k] = np.nanstd(diff_matrix[:, k])

    plt.bar(x_position[k], means[k], color=colors[k], yerr=stds[k], align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.text(x_position[k], 0.01, str(round(means[k], 2)), color='black', fontsize=10)

plt.xlabel('bandits', fontweight='bold')
plt.ylabel('final value - initial value in p1', fontweight='bold')
plt.grid('silver', linestyle='-', linewidth=0.4)
plt.savefig('value_diff_per_participant_bandit_M' + str(model) + '_bars.png', bbox_inches='tight')
plt.close()