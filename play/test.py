import os
import numpy as np
import matplotlib.pyplot as plt

from ConfLearning.tasks.two_armed_bandit import TwoArmedBandit
from ConfLearning.models.rl_simple import Rescorla, RescorlaZero, RescorlaConf, RescorlaConfGen, RescorlaConfBase, RescorlaConfBaseGen, RescorlaConfZero, RescorlaConfZeroGen, RescorlaConfBaseZero, RescorlaConfBaseZeroGen, BayesModel, BayesIdealObserver

# os.makedirs('../figures/simulation')

niterate = 100
ntrials = 100
nmodels = 12
nbandits = 5

values = np.array([0.2, 0.4, 0.5, 0.6, 0.8])

task = TwoArmedBandit(values=values, nbandits=nbandits)
models = [Rescorla(), RescorlaZero(), RescorlaConf(), RescorlaConfGen(), RescorlaConfBase(), RescorlaConfBaseGen(), RescorlaConfZero(), RescorlaConfZeroGen(), RescorlaConfBaseZero(), RescorlaConfBaseZeroGen(), BayesModel(), BayesIdealObserver()]

performance = np.full((nmodels, niterate, ntrials), np.nan)
performance_percent = np.full((nmodels, ntrials), np.nan)

new_values_choice = np.full((nmodels, niterate, ntrials, nbandits), 0, float)

for i in range(niterate):
    for m, model in enumerate(models):
        for t in range(ntrials):

            model.get_current_trial(t)

            stims = task.get_stimuli()
            model.stims = stims

            model.get_choice_probab()
            choice = model.choice()

            if (1 <= m <= 3) and (t >= 30):
                outcome = float("NaN")
            else:
                outcome = task.get_outcome(choice)

            new_value_choice = model.update(outcome, model.get_confidence())

            for n in range(nbandits):

                if n == choice:
                    new_values_choice[m, i, t, n] = new_value_choice
                else:
                    new_values_choice[m, i, t, n] = new_values_choice[m, i, t - 1, n]

            if np.max(values[stims]) == values[choice]:
                performance[m, i, t] = 1
            else:
                performance[m, i, t] = 0

for o in range(nmodels):
    for j in range(ntrials):
        performance_percent[o, j] = 100 * len([True for x in performance[o, :, j] if x == 1]) / len(performance[o, :, j])

for o, on in enumerate(models):
    for k in range(nbandits):

        locals()["bandit" + str(k) + "model" + str(o)] = np.mean(new_values_choice[o, :, :, k], axis=0)

    plt.plot(range(ntrials), eval("bandit0model" + str(o)), 'r--', range(ntrials), eval("bandit1model" + str(o)), 'b--',
             range(ntrials), eval("bandit2model" + str(o)), 'g--', range(ntrials), eval("bandit3model" + str(o)), 'y--',
             range(ntrials), eval("bandit4model" + str(o)), 'm--')

    plt.title('learning curve - ' + str(on)[31:-30])
    plt.xlabel('trials', fontweight='bold')
    plt.ylabel('simulation of learned bandit value)', fontweight='bold')
    plt.savefig('../figures/simulation/model' + str(o) + '.png', bbox_inches='tight')

    # plt.scatter(range(ntrials), performance_percent[o, :])

    plt.close()
