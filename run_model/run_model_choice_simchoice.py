import os
import numpy as np
import pandas as pd

from ConfLearning.models.rl_simple import Rescorla, RescorlaZero as RescorlaDeval
from ConfLearning.models.rl_simple_choice_simchoice import RescorlaChoiceMono, RescorlaChoiceDual, RescorlaChoiceMonoDeval, RescorlaChoiceDualDeval
from ConfLearning.models.maximum_likelihood import ParameterFit
from pathlib import Path

fitting = ParameterFit()

cwd = Path.cwd()
# print(cwd)
path_data = os.path.join(cwd, '../data/')

# os.makedirs('../results/fittingData')
# os.makedirs('../results/choiceProbab')

matrix = pd.read_pickle(os.path.join(path_data, 'data.pkl'))

var_list = ['stim_left', 'stim_right', 'chosen_stim', 'outcome_value', 'correct_value', 'true_value']
stim_left, stim_right, chosen_stim, outcome_value, correct_value, true_value = None, None, None, None, None, None

for v, variable in enumerate(var_list):
    locals()[variable] = np.load(os.path.join(path_data, variable + '.npy'))

# set_model = 4   # CHANGE HERE

nsubjects = max(matrix.subject.values) + 1
nblocks = max(matrix.block.values) + 1
nphases = max(matrix[~matrix.phase.isna()].phase.values) + 1
# ntrials = 56
ntrials_phase_max = 18
nbandits = 5

alpha = 0.1
beta = 1
alpha_n = 0.1
gamma = 0.1

la, ua = 0, 1
lb, ub = 0, 2
lan, uan = 0, 1
lg, ug = 0, 100

grid_alpha = np.arange(0.1, 0.51, 0.2)
grid_alpha_n = np.arange(0.01, 0.061, 0.05)
grid_gamma = np.hstack((0, np.arange(0.05, 0.5, 0.05)))    # np.arange(0.05, 0.5, 0.1)
grid_beta = np.arange(0.1, 0.31, 0.1)

bounds = [np.c_[np.array([la, lb]), np.array([ua, ub])],
          np.c_[np.array([la, lb, lan]), np.array([ua, ub, uan])],
          *[np.c_[np.array([la, lb, lg]), np.array([ua, ub, ug])] for _ in range(2)],
          *[np.c_[np.array([la, lb, lg, lan]), np.array([ua, ub, ug, uan])] for _ in range(2)]
]

expect = [(np.array([ua, ub]) - np.array([la, lb])) / 2,
          (np.array([ua, ub, uan]) - np.array([la, lb, lan])) / 2,
          *[(np.array([ua, ub, ug]) - np.array([la, lb, lg])) / 2 for _ in range(2)],
          *[(np.array([ua, ub, ug, uan]) - np.array([la, lb, lg, lan])) / 2 for _ in range(2)]
]

grid_range = [
    [grid_alpha, grid_beta],
    [grid_alpha, grid_beta, grid_alpha_n],
    *[[grid_alpha, grid_beta, grid_gamma] for _ in range(2)],
    *[[grid_alpha, grid_beta, grid_gamma, grid_alpha_n] for _ in range(2)]
]

# expect = [[0.1, 1], [0.1, 1, 0], [0.1, 1, 0, 0], [0.1, 1, 0, 0], [0.1, 1, 0, 0], [0.1, 1, 0, 0], [0.1, 1, 0, 0]]

modellist = [Rescorla, RescorlaDeval, RescorlaChoiceMono, RescorlaChoiceDual, RescorlaChoiceMonoDeval, RescorlaChoiceDualDeval]
model_names = ['Static', 'Deval', 'Mono', 'Dual', 'MonoDeval', 'DualDeval']
paramlist = [[alpha, beta], [alpha, beta, alpha_n], *[[alpha, beta, gamma] for _ in range(2)], *[[alpha, beta, gamma, alpha_n] for _ in range(2)]]
nparams = [2, 3, 3, 3, 4, 4]
# modellist = [RescorlaChoiceMono]
# model_names = ['Mono']
# paramlist = [[alpha, beta, gamma]]
# nparams = [3]
nmodels = len(modellist)

probab_choice = np.full((nmodels, nsubjects, nblocks, nphases, ntrials_phase_max), np.nan)
paramfit = np.full((nmodels, nsubjects, max(nparams)), np.nan)
negll = np.full((nmodels, nsubjects), np.nan)
AIC, BIC = np.full((nmodels, nsubjects), np.nan, float), np.full((nmodels, nsubjects), np.nan)

fittingParams, fittingModel, saveChoiceProbab = None, None, None


def run_model(params, modelspec, s, return_cp=False, return_full=False, return_nll=False):

    model = modelspec(*params)

    negLogL = 0

    if return_nll:
        negLoLi = np.full((nblocks, nphases, ntrials_phase_max), np.nan, float)

    if return_cp:
        choiceprob = np.full((nblocks, nphases, ntrials_phase_max), np.nan)

    if return_full:
        new_values_choice = np.full((nblocks, nphases, ntrials_phase_max, nbandits), np.nan, float)
        true_values_choice = np.full((nblocks, nbandits), np.nan, float)
        performance = np.full((nblocks, nphases, ntrials_phase_max), np.nan, float)

    for b in range(nblocks):

        model.values = np.full(nbandits, 0, float)

        for p in range(nphases):
            for i, t in enumerate(np.where(~np.isnan(stim_left[s, b, p]))[0]):

                model.get_current_trial(t)

                model.stims = np.array([int(stim_left[s, b, p, t]), int(stim_right[s, b, p, t])])
                model.stim_chosen = int(chosen_stim[s, b, p, t])

                cp = model.get_choice_probab()
                model.predicted_choice()

                if return_cp:
                    choiceprob[b, p, i] = cp

                negLogL -= np.log(np.maximum(cp, 1e-8))

                if return_nll:
                    negLoLi[b, p, i] = negLogL

                # confidence = 2 * cp - 1 if cp >= 0.5 else 2 * (1 - cp) - 1
                # new_value_choice = model.update(outcome_value[s, b, p, t], confidence_value[s, b, p, t])


                confidence = model.get_confidence()

                new_value_choice = model.update(outcome_value[s, b, p, t], confidence)

                if return_full:
                    performance[b, p, i] = 0 if (correct_value[s, b, p, t] == False) else 1

                    for k in range(nbandits):

                        if k == model.stim_chosen:
                            new_values_choice[b, p, i, k] = new_value_choice
                        else:
                            if (p > 0) & (i == 0):
                                if np.all(np.isnan(new_values_choice[b, p - 1, :, k])):
                                    new_values_choice[b, p, i, k] = new_values_choice[b, p-2 , :, k][~np.isnan(new_values_choice[b, p-2, :, k])][-1]
                                else:
                                    new_values_choice[b, p, i, k] = new_values_choice[b, p - 1, :, k][~np.isnan(new_values_choice[b, p - 1, :, k])][-1]
                            else:
                                new_values_choice[b, p, i, k] = 0 if t == 0 else new_values_choice[b, p, i - 1, k]

        if return_full:
            true_values_choice[b, :] = true_value[s, b, :]

    if return_nll:
        return negLoLi

    if return_full == False:
        return (negLogL, choiceprob) if return_cp else negLogL
    else:
        return new_values_choice, true_values_choice, performance


if __name__ == '__main__':

    set_model = -1

    for m, models in enumerate(modellist):
        print(f'Model {m+1} / {len(modellist)}')
        for n in range(nsubjects):
            if models != modellist[set_model]:
                print(f'[Model {m+1} / {len(modellist)}] Fitting model {modellist[set_model]}')
            #     continue

                params = paramlist[m]
                fitting.set_model(n, nsubjects, modellist[m], run_model, nparams[m])
                fitting.local_minima(expect[m], bounds[m], grid_range[m])
                print(f'\tSubject {n+1} / {nsubjects}')

                for param in range(nparams[m]):
                    # paramfit[m, n, param] = fitting.data[n, param]
                    # when unbounded optimization is used, we apply bounds post-hoc:
                    paramfit[m, n, param] = min(bounds[m][param][1], max(bounds[m][param][0], fitting.data[n, param]))

                negll[m, n], probab_choice[m, n] = run_model(fitting.data[n], modellist[m], n, return_cp=True, return_full=False)
                nsamples = np.sum(~np.isnan(probab_choice[m, n]))

                AIC[m, n], BIC[m, n] = fitting.model_fit(negll[m, n], nsamples)

                locals()["parameter_m" + str(m)] = pd.DataFrame(data={"ALPHA": paramfit[m, :, 0], "BETA": paramfit[m, :, 1],
                                                                      "GAMMA": paramfit[m, :, 2], "ALPHA_N": paramfit[m, :, 3]},
                                                                columns=["ALPHA", "BETA", "GAMMA", "ALPHA_N"])
                locals()["fit_m" + str(m)] = pd.DataFrame(data={"AIC": AIC[m, :], "BIC": BIC[m, :], "NEGLL": negll[m, :]},
                                                          columns=["AIC", "BIC", "NEGLL"])
                locals()["choice_probab_m" + str(m)] = pd.DataFrame(data={"choice_probab_subj" + str(n): probab_choice[m, n][~np.isnan(probab_choice[m, n])]},
                                                                    columns=["choice_probab_subj" + str(n)])
                saveChoiceProbab = pd.concat([saveChoiceProbab, eval("choice_probab_m" + str(m))], axis=1)
                locals()["param_corr_m" + str(m)] = eval("parameter_m" + str(m)).corr()

            pd.concat([eval("parameter_m" + str(m)), eval("fit_m" + str(m))], axis=1).to_pickle(f"../results/fittingData/fittingData_{model_names[m]}_choice_simchoice.pkl", protocol=4)
        # saveChoiceProbab.to_pickle("../results/choiceProbab/choiceProbabM" + str(m) + "_choice.pkl", protocol=4)
