import numpy as np
import matplotlib.pyplot as plt

from tasks.two_armed_bandit import TwoArmedBandit
from models.rl_simple import RLModel, ConfidencePE, ConfidencePEgeneric, ConfidenceIdealObserver, BayesModel, BayesIdealObserver

__all__ = [RLModel, ConfidencePE, ConfidencePEgeneric, ConfidenceIdealObserver, BayesModel, BayesIdealObserver]

niterate = 100
ntrials = 100
nmodels = 6

bandit0model3 = None
bandit1model3 = None
bandit2model3 = None
bandit3model3 = None
bandit4model3 = None

nbandits = 5
values = np.array([0.2, 0.4, 0.5, 0.6, 0.8])

task = TwoArmedBandit(values=values, nbandits=nbandits)
models = [RLModel(), ConfidencePE(), ConfidencePEgeneric(), ConfidenceIdealObserver(), BayesModel(), BayesIdealObserver()]

performance = np.full((nmodels, niterate, ntrials), np.nan)
performance_percent = np.full((nmodels, ntrials), np.nan)

new_values_choice = np.full((nmodels, niterate, ntrials, nbandits), 0, float)

for i in range(niterate):
    for m, model in enumerate(models):
        for t in range(ntrials):

            model.get_current_trial(t)

            stims = task.get_stimuli()
            model.set_stimuli(stims)

            model.get_choice_probab()
            choice = model.choice()

            if (1 <= m <= 3) and (t >= 30):
                outcome = float("NaN")
            else:
                outcome = task.get_outcome(choice)

            confidence = model.get_confidence()
            new_value_choice = model.update(outcome, confidence)

            for n in range(nbandits):

                if n == choice:
                    new_values_choice[m, i, t, n] = new_value_choice
                else:
                    new_values_choice[m, i, t, n] = new_values_choice[m, i, t - 1, n]

            if np.max(values[stims]) == values[choice]:
                performance[m, i, t] = 1
            else:
                performance[m, i, t] = 0

for j in range(ntrials):
    for o in range(nmodels):
        performance_percent[o, j] = 100 * len([True for x in performance[o, :, j] if x == 1]) / len(performance[o, :, j])

for k in range(nbandits):
    for o in range(nmodels):
        locals()["bandit" + str(k) + "model" + str(o)] = np.mean(new_values_choice[o, :, :, k], axis=0)

plt.plot(range(ntrials), bandit0model3, 'r--', range(ntrials), bandit1model3, 'b--',
         range(ntrials), bandit2model3, 'g--', range(ntrials), bandit3model3, 'y--',
         range(ntrials), bandit4model3, 'm--')
# plt.scatter(range(ntrials), performance_percent[3, :])
plt.show()
