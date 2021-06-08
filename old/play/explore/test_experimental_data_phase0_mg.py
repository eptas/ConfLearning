import os
import numpy as np
import pandas as pd

from ConfLearning.models.rl_simple import Rescorla, RescorlaZero, RescorlaConf, RescorlaConfGen, RescorlaConfBase, RescorlaConfBaseGen, RescorlaConfZero, RescorlaConfZeroGen, RescorlaConfBaseZero, RescorlaConfBaseZeroGen, BayesModel, BayesIdealObserver, RescorlaConfGamma, RescorlaConfGenGamma
from ConfLearning.fitting.maximum_likelihood import ParameterFit
from pathlib import Path

fitting = ParameterFit()

cwd = Path.cwd()
# print(cwd)
path_data = os.path.join(cwd, '../data/')

# os.makedirs('../results/fittingData')
# os.makedirs('../results/choiceProbab')

matrix = pd.read_pickle(os.path.join(path_data, 'data.pkl'))

var_list = ['stim_left', 'stim_right', 'chosen_stim', 'outcome_value', 'confidence_value', 'correct_value', 'true_value']
stim_left, stim_right, chosen_stim, outcome_value, confidence_value, correct_value, true_value = None, None, None, None, None, None, None

for v, variable in enumerate(var_list):
    locals()[variable] = np.load(os.path.join(path_data, variable + '.npy'))

nsubjects = max(matrix.subject.values) + 1
nblocks = max(matrix.block.values) + 1
nphases = 1
# ntrials = 56
ntrials_phase_max = 18
nbandits = 5

alpha = 0.1
beta = 1

lower_alpha, la = 0, 0
lower_beta, lb = 0, 0
upper_alpha, ua = 1, 1
upper_beta, ub = 2, 2

grid_alpha = np.arange(0.1, 0.51, 0.2)
grid_beta = np.arange(0.1, 0.31, 0.1)

bounds = np.c_[np.array([la, lb]), np.array([ua, ub])]

expect = (np.array([ua, ub]) - np.array([la, lb])) / 2

grid_range = [grid_alpha, grid_beta]

model = Rescorla
phase = 0
paramlist = [alpha, beta]
nparams = 2

probab_choice = np.full((nsubjects, nblocks, ntrials_phase_max), np.nan)
paramfit = np.full((nsubjects, nparams), np.nan)
negll = np.full(nsubjects, np.nan)
AIC, BIC = np.full(nsubjects, np.nan, float), np.full(nsubjects, np.nan)

fittingParams, fittingModel, saveChoiceProbab = None, None, None


def run_model(params, modelspec, s, return_cp=False):

    model = modelspec(*params)

    negLogL = 0

    if return_cp:
        choiceprob = np.full((nblocks, ntrials_phase_max), np.nan)

    for b in range(nblocks):

        model.values = np.full(nbandits, 0, float)

        for i, t in enumerate(np.where(~np.isnan(stim_left[s, b, phase]))[0]):

            model.get_current_trial(t)

            model.stims = np.array([int(stim_left[s, b, phase, t]), int(stim_right[s, b, phase, t])])
            model.stim_chosen = int(chosen_stim[s, b, phase, t])

            cp = model.get_choice_probab()

            if return_cp:
                choiceprob[b, i] = cp

            negLogL -= np.log(np.maximum(cp, 1e-8))

            new_value_choice = model.update(outcome_value[s, b, phase, t])

    return (negLogL, choiceprob) if return_cp else negLogL


if __name__ == '__main__':

    for n in range(nsubjects):

        # params = paramlist
        fitting.set_model(n, nsubjects, model, run_model, nparams)
        fitting.local_minima(expect, bounds, grid_range, grid_multiproc=False)
        print(n)

        for param in range(nparams):
            # paramfit[m, n, param] = fitting.data[n, param]
            # when unbounded optimization is used, we apply bounds post-hoc:
            paramfit[n, param] = min(bounds[param][1], max(bounds[param][0], fitting.data[n, param]))

        negll[n], probab_choice[n] = run_model(fitting.data[n], model, n, return_cp=True)
        nsamples = np.sum(~np.isnan(probab_choice[n]))

        AIC[n], BIC[n] = fitting.model_fit(negll[n], nsamples)

        locals()["parameter_phase0"] = pd.DataFrame(data={"ALPHA": paramfit[:, 0], "BETA": paramfit[:, 1]},
                                                        columns=["ALPHA", "BETA"])
        locals()["fit_phase0"] = pd.DataFrame(data={"AIC": AIC[:], "BIC": BIC[:], "NEGLL": negll[:]},
                                                  columns=["AIC", "BIC", "NEGLL"])
        locals()["choice_probab_phase0"] = pd.DataFrame(data={"choice_probab_subj" + str(n): probab_choice[n][~np.isnan(probab_choice[n])]},
                                                            columns=["choice_probab_subj" + str(n)])
        saveChoiceProbab = pd.concat([saveChoiceProbab, eval("choice_probab_phase0")], axis=1)
        locals()["param_corr_phase0"] = eval("parameter_phase0").corr()

    pd.concat([eval("parameter_phase0"), eval("fit_phase0")], axis=1).to_pickle(f"../results/fittingData/fittingData_phase0.pkl", protocol=4)
    saveChoiceProbab.to_pickle(f"../results/choiceProbab/choiceProbab_phase0.pkl", protocol=4)
