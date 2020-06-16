import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from play.test_experimental_data import run_model, matrix, modellist, nsubjects, nbandits

model = 6

fittingData = pd.read_pickle('C:/Users/esthe/PycharmProjects/MetaCognition/play/fittingDataM' + str(model) + '.pkl', compression=None)
choiceProbab = pd.read_pickle('C:/Users/esthe/PycharmProjects/MetaCognition/play/choiceProbabM' + str(model) + '.pkl', compression=None)

alpha = fittingData.ALPHA
beta = fittingData.BETA
alpha_c = fittingData.ALPHA_C   # HERE PHI & ALPHA_N AS WELL
gamma = fittingData.GAMMA

learned_value0, learned_value1, learned_value2, learned_value3, learned_value4 = None, None, None, None, None
learned_value_per_phase0, learned_value_per_phase1, learned_value_per_phase2, learned_value_per_phase3, learned_value_per_phase4 = None, None, None, None, None

value0, value1, value2, value3, value4 = None, None, None, None, None
phase_value0, phase_value1, phase_value2, phase_value3, phase_value4 = None, None, None, None, None
mean_value0, mean_value1, mean_value2, mean_value3, mean_value4 = None, None, None, None, None

performance_matrix = None

x_axis = range(33)
colors = ['r', 'b', 'g', 'y', 'm']


for m, models in enumerate(modellist):
    for n in range(nsubjects):
        if models != modellist[model]:
            continue

        parameter = [[alpha[n], beta[n]], [alpha[n], beta[n], alpha_c[n]], [alpha[n], beta[n], alpha_c[n], gamma[n]], [alpha[n], beta[n], alpha_c[n], gamma[n]],
                     [alpha[n], beta[n], alpha_c[n], gamma[n]], [alpha[n], beta[n], alpha_c[n], gamma[n]], [alpha[n], beta[n], alpha_c[n], gamma[n]]]

        new_value_choice, true_value_choice, performance = run_model(models, parameter[m], n, return_full=True)

        performances = pd.DataFrame(data={"s" + str(n): performance[:, :, :][~np.isnan(performance[:, :, :])]},
                                    columns=["s" + str(n)])
        performance_matrix = pd.concat([performance_matrix, performances], axis=1)

        for k in range(nbandits):
            for b in range(max(matrix.block.values) + 1):

                learn_value = pd.DataFrame(data={"s" + str(n) + "b" + str(b): true_value_choice[b, k] - new_value_choice[b, :, :, k][~np.isnan(new_value_choice[b, :, :, k])]},
                                           columns=["s" + str(n) + "b" + str(b)])
                locals()["learned_value" + str(k)] = pd.concat([eval("learned_value" + str(k)), learn_value], axis=1)

                for p in range(max(matrix.phase.values) + 1):

                    phase = pd.DataFrame(data={"s" + str(n) + "b" + str(b) + "p" + str(p): new_value_choice[b, p, :, k][~np.isnan(new_value_choice[b, p, :, k])]},
                                         columns=["s" + str(n) + "b" + str(b) + "p" + str(p)])
                    locals()["learned_value_per_phase" + str(k)] = pd.concat([eval("learned_value_per_phase" + str(k)), phase], axis=1)

performance_percent = 100 * performance_matrix[performance_matrix == 1].count(axis=1) / performance_matrix.count(axis=1)

for k in range(nbandits):
    for b in range(max(matrix.block.values) + 1):

        block_values = eval("learned_value" + str(k)).filter(regex="b" + str(b)).mean(axis=1)
        locals()["value" + str(k)] = np.append(eval("value" + str(k)), block_values[~np.isnan(block_values)])

        for p in range(max(matrix.phase.values) + 1):

            phase_values = eval("learned_value_per_phase" + str(k)).filter(regex="b" + str(b) + "p" + str(p)).mean(axis=0)
            locals()["phase_value" + str(k)] = np.append(eval("phase_value" + str(k)), np.nanmean(phase_values))

    for p in range(max(matrix.phase.values) + 1):

        mean_values = eval("learned_value_per_phase" + str(k)).filter(regex="p" + str(p)).mean(axis=1)
        locals()["mean_value" + str(k)] = np.append(eval("mean_value" + str(k)), mean_values[~np.isnan(mean_values)])


plt.figure(0)

plt.plot(range(len(value0)), value0, 'r', range(len(value1)), value1, 'b', range(len(value2)), value2, 'g',
         range(len(value3)), value3, 'y', range(len(value4)), value4, 'm', linewidth=0.5)
plt.xlabel('trials')
plt.ylabel('true bandit value - learned value')
plt.title('learning curve - ' + str(modellist[model])[25:-2])
plt.xlim(0, len(value0))
plt.ylim((-31, 41))
plt.grid('silver', linestyle='-', linewidth=0.4)
plt.savefig('M' + str(model) + '.png', bbox_inches='tight')

plt.figure(1)

for k in range(nbandits):

    array = pd.Series(eval("phase_value" + str(k))[1:len(eval("phase_value" + str(k)))]).fillna(method='ffill')
    plt.plot(x_axis, array, color=colors[k], linewidth=0.5)

plt.xlabel('blocks')
plt.ylabel('learned bandit value ($real_{min}= 12; real_{max}= 40$)')
plt.title('learned values per block & phase - ' + str(modellist[model])[25:-2])
plt.xticks(np.arange(len(x_axis)), (max(matrix.block.values) + 1) * ['0', '1', '2'], fontsize=6)
plt.ylim((0, 31))
plt.grid('silver', linestyle='-', linewidth=0.4)
plt.savefig('M' + str(model) + '_per_phase' + '.png', bbox_inches='tight')

plt.figure(2)

for i in range(nbandits):

    x, y = range(len(eval("mean_value" + str(i))) - 1), eval("mean_value" + str(i))[1:len(eval("mean_value" + str(i)))]
    plt.plot(x, y, color=colors[i], linewidth=0.5)

plt.xlabel('trials')
plt.ylabel('learned bandit value ($real_{min}= 12; real_{max}= 40$)')
plt.title('learning curve across blocks - ' + str(modellist[model])[25:-2])
plt.ylim((0, 31))
plt.xlim(0, len(mean_value0))
plt.grid('silver', linestyle='-', linewidth=0.4)
plt.savefig('M' + str(model) + '_across_blocks' + '.png', bbox_inches='tight')

plt.figure(3)

plt.scatter(range(len(performance_percent)), performance_percent)
plt.show()
