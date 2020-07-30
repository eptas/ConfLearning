import os
import numpy as np
import pandas as pd

from models.rl_simple import Rescorla, RescorlaZero, RescorlaConf, RescorlaConfGen, RescorlaConfBase, RescorlaConfBaseGen, RescorlaConfWeighted, RescorlaConfWeightedGen, RescorlaConfAll, RescorlaConfAllGen, BayesModel, BayesIdealObserver
from fitting.maximum_likelihood import ParameterFit
from pathlib import Path

fitting = ParameterFit()

cwd = Path.cwd()
path_data = os.path.join(cwd, '../data/')

matrix = pd.read_pickle(os.path.join(path_data, 'data.pkl'))

var_list = ['stim_left', 'stim_right', 'chosen_stim', 'outcome_value', 'confidence_value', 'correct_value', 'true_value']
stim_left, stim_right, chosen_stim, outcome_value, confidence_value, correct_value, true_value = None, None, None, None, None, None, None

for v, variable in enumerate(var_list):
    locals()[variable] = np.load(os.path.join(path_data, variable + '.npy'))

set_model = 5   # CHANGE HERE

nsubjects = max(matrix.subject.values) + 1
nblocks = max(matrix.block.values) + 1
nphases = max(matrix[~matrix.phase.isna()].phase.values) + 1
# ntrials = 56
ntrials_phase_max = 18
nbandits = 5

alpha = 0.1
alpha_n = 0.1
alpha_c = 0.1
beta = 3
gamma = 0.1
phi = 0.1

lower_alpha, la = 0, 0
lower_alpha_n, lan = 0, 0
lower_alpha_c, lac = 0, 0
lower_beta, lb = 0, 0
lower_gamma, lg = 0, 0
lower_phi, lp = 0, 0

upper_alpha, ua = 1, 1
upper_alpha_n, uan = 1, 1
upper_alpha_c, uac = 1, 1
upper_beta, ub = 10, 10
upper_gamma, ug = 1, 1
upper_phi, up = 1, 1

bounds = [np.c_[np.array([la, lb]), np.array([ua, ub])],
          np.c_[np.array([la, lb, lan]), np.array([ua, ub, uan])],
          *[np.c_[np.array([la, lb, lac, lg]), np.array([ua, ub, uac, ug])] for _ in range(6)],
          *[np.c_[np.array([lb, lac, lg]), np.array([ub, uac, ug])] for _ in range(2)],
          *[np.c_[np.array([la, lb, lp, lg]), np.array([ua, ub, up, ug])] for _ in range(2)]]

expect = [(np.array([ua, ub]) - np.array([la, lb])) / 2,
          (np.array([ua, ub, uan]) - np.array([la, lb, lan])) / 2,
          *[(np.array([ua, ub, uac, ug]) - np.array([la, lb, lac, lg])) / 2 for _ in range(6)],
          *[(np.array([ub, uac, ug]) - np.array([lb, lac, lg])) / 2 for _ in range(2)],
          *[(np.array([ua, ub, up, ug]) - np.array([la, lb, lp, lg])) / 2 for _ in range(2)]]

# expect = [[0.1, 1], [0.1, 1, 0], [0.1, 1, 0, 0], [0.1, 1, 0, 0], [0.1, 1, 0, 0], [0.1, 1, 0, 0], [0.1, 1, 0, 0]]

modellist = [Rescorla, RescorlaZero, RescorlaConf, RescorlaConfGen, RescorlaConfBase, RescorlaConfBaseGen, RescorlaConfWeighted, RescorlaConfWeightedGen, RescorlaConfAll, RescorlaConfAllGen, BayesModel, BayesIdealObserver]
paramlist = [[alpha, beta], [alpha, beta, alpha_n], *[[alpha, beta, alpha_c, gamma] for _ in range(6)], *[[beta, alpha_c, gamma] for _ in range(2)], *[[alpha, beta, phi, gamma] for _ in range(2)]]
nparams = [2, 3, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4]
nmodels = len(modellist)

probab_choice = np.full((nmodels, nsubjects, nblocks, nphases, ntrials_phase_max), np.nan)
paramfit = np.full((nmodels, nsubjects, max(nparams)), np.nan)
negll = np.full((nmodels, nsubjects), np.nan)
AIC, BIC = np.full((nmodels, nsubjects), np.nan, float), np.full((nmodels, nsubjects), np.nan)

fittingParams, fittingModel, saveChoiceProbab = None, None, None


def run_model(params, modelspec, s, return_cp=False, return_full=False):

    model = modelspec(*params)

    negLogL = 0

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

                if return_cp:
                    choiceprob[b, p, i] = cp

                negLogL -= np.log(np.maximum(cp, 1e-8))
                new_value_choice = model.update(outcome_value[s, b, p, t], confidence_value[s, b, p, t])

                if return_full:
                    performance[b, p, i] = 0 if (correct_value[s, b, p, t] == False) else 1

                    for k in range(nbandits):

                        if k == model.stim_chosen:
                            new_values_choice[b, p, i, k] = new_value_choice
                        else:
                            new_values_choice[b, p, i, k] = 0 if t == 0 else new_values_choice[b, p, i - 1, k]

        if return_full:
            true_values_choice[b, :] = true_value[s, b, :]

    if return_full == False:
        return (negLogL, choiceprob) if return_cp else negLogL
    else:
        return new_values_choice, true_values_choice, performance


if __name__ == '__main__':

    for m, models in enumerate(modellist):
        for n in range(nsubjects):
            if models != modellist[set_model]:
                continue

            params = paramlist[m]
            fitting.set_model(n, nsubjects, modellist[m], run_model, nparams[m])
            fitting.local_minima(expect[m], bounds[m])
            print(n)

            for param in range(nparams[m]):
                # paramfit[m, n, param] = fitting.data[n, param]
                # when unbounded optimization is used, we apply bounds post-hoc:
                paramfit[m, n, param] = min(bounds[m][param][1], max(bounds[m][param][0], fitting.data[n, param]))

            negll[m, n], probab_choice[m, n] = run_model(fitting.data[n], modellist[m], n, return_cp=True, return_full=False)
            nsamples = np.sum(~np.isnan(probab_choice[m, n]))

            AIC[m, n], BIC[m, n] = fitting.model_fit(negll[m, n], nsamples)

            locals()["parameter_m" + str(m)] = pd.DataFrame(data={"ALPHA": paramfit[m, :, 0], "BETA": paramfit[m, :, 1],
                                                                  "ALPHA_C": paramfit[m, :, 2], "GAMMA": paramfit[m, :, 3]},
                                                            columns=["ALPHA", "BETA", "ALPHA_C", "GAMMA"])
            locals()["fit_m" + str(m)] = pd.DataFrame(data={"AIC": AIC[m, :], "BIC": BIC[m, :], "NEGLL": negll[m, :]},
                                                      columns=["AIC", "BIC", "NEGLL"])
            locals()["choice_probab_m" + str(m)] = pd.DataFrame(data={"choice_probab_subj" + str(n): probab_choice[m, n][~np.isnan(probab_choice[m, n])]},
                                                                columns=["choice_probab_subj" + str(n)])
            saveChoiceProbab = pd.concat([saveChoiceProbab, eval("choice_probab_m" + str(m))], axis=1)
            locals()["param_corr_m" + str(m)] = eval("parameter_m" + str(m)).corr()


# pd.concat([eval("parameter_m" + str(set_model)), eval("fit_m" + str(set_model))], axis=1).to_pickle("../results/fittingDataM" + str(set_model) + ".pkl")
# saveChoiceProbab.to_pickle("../results/choiceProbabM" + str(set_model) + ".pkl")
