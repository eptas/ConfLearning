import numpy as np
import pandas as pd

from models.rl_simple import RLModel, RLModelWithoutFeedback, ConfidencePE, ConfidencePEgeneric, ConfidenceIdealObserver, BayesModel, BayesIdealObserver
from fitting.maximum_likelihood import ParameterFit

fitting = ParameterFit()

matrix = pd.read_pickle('C:/Users/esthe/Desktop/data.pkl', compression=None)

stim_left = np.load('C:/Users/esthe/PycharmProjects/MetaCognition/data/stim_left.npy')
stim_right = np.load('C:/Users/esthe/PycharmProjects/MetaCognition/data/stim_right.npy')
chosen_stim = np.load('C:/Users/esthe/PycharmProjects/MetaCognition/data/chosen_stim.npy')
outcome_value = np.load('C:/Users/esthe/PycharmProjects/MetaCognition/data/outcome_value.npy')
confidence_value = np.load('C:/Users/esthe/PycharmProjects/MetaCognition/data/confidence_value.npy')
correct_value = np.load('C:/Users/esthe/PycharmProjects/MetaCognition/data/correct_value.npy')
true_value = np.load('C:/Users/esthe/PycharmProjects/MetaCognition/data/true_value.npy')

set_model = 4   # CHANGE HERE

nsubjects = max(matrix.subject.values)
nblocks = max(matrix.block.values) + 1
nphases = max(matrix[~matrix.phase.isna()].phase.values) + 1
ntrials = 56
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
upper_beta, ub = 100, 100
upper_gamma, ug = 1, 1
upper_phi, up = 1, 1

bounds = [np.c_[np.array([la, lb]), np.array([ua, ub])],
          np.c_[np.array([la, lb, lan]), np.array([ua, ub, uan])],
          np.c_[np.array([la, lb, lac, lg]), np.array([ua, ub, uac, ug])],
          np.c_[np.array([la, lb, lac, lg]), np.array([ua, ub, uac, ug])],
          np.c_[np.array([la, lb, lac, lg]), np.array([ua, ub, uac, ug])],
          np.c_[np.array([la, lb, lp, lg]), np.array([ua, ub, up, ug])],
          np.c_[np.array([la, lb, lp, lg]), np.array([ua, ub, up, ug])]]

expect = [(np.array([ua, ub]) - np.array([la, lb])) / 2,
          (np.array([ua, ub, uan]) - np.array([la, lb, lan])) / 2,
          (np.array([ua, ub, uac, ug]) - np.array([la, lb, lac, lg])) / 2,
          (np.array([ua, ub, uac, ug]) - np.array([la, lb, lac, lg])) / 2,
          (np.array([ua, ub, uac, ug]) - np.array([la, lb, lac, lg])) / 2,
          (np.array([ua, ub, up, ug]) - np.array([la, lb, lp, lg])) / 2,
          (np.array([ua, ub, up, ug]) - np.array([la, lb, lp, lg])) / 2]

modellist = [RLModel, RLModelWithoutFeedback, ConfidencePE, ConfidencePEgeneric, ConfidenceIdealObserver, BayesModel, BayesIdealObserver]
paramlist = [[alpha, beta], [alpha, beta, alpha_n], [alpha, beta, alpha_c, gamma], [alpha, beta, alpha_c, gamma], [alpha, beta, alpha_c, gamma], [alpha, beta, phi, gamma], [alpha, beta, phi, gamma]]
nparams = [2, 3, 4, 4, 4, 4, 4]

probab_choice = np.full((nsubjects, len(modellist), (ntrials * nblocks)), np.nan, float)
paramfit = np.full((nsubjects, len(modellist), max(nparams)), np.nan, float)
negll = np.full((nsubjects, len(modellist)), np.nan, float)
AIC, BIC = np.full((nsubjects, len(modellist)), np.nan, float), np.full((nsubjects, len(modellist)), np.nan, float)

fittingParams, fittingModel, saveChoiceProbab = None, None, None


def run_model(modelspec, params, s, return_full=False):

    model = modelspec(*params)

    choice_probab = np.array([])
    new_values_choice = np.full((nblocks, nphases, ntrials, nbandits), np.nan, float)
    true_values_choice = np.full((nblocks, nbandits), np.nan, float)
    performance = np.full((nblocks, nphases, ntrials), np.nan, float)

    for index, b in enumerate(matrix[(matrix.subject == s)].block.unique()):

        model.values = np.full(nbandits, 0, float)

        for ind, p in enumerate(matrix[(matrix.subject == s) & (matrix.block == b) & (~matrix[~matrix.phase.isna()].phase.isna())].phase.unique()):
            for i, t in enumerate(matrix[(matrix.subject == s) & (matrix.block == b) & (matrix[~matrix.phase.isna()].phase == p) & matrix.type_choice_obs].trial.values):

                model.get_current_trial(t)

                stims = np.array([int(stim_left[s, b, p, t]), int(stim_right[s, b, p, t])])
                model.set_stimuli(stims)

                probab = model.get_choice_probab()
                choice_probab = np.append(choice_probab, probab)

                choice = int(chosen_stim[s, b, p, t])
                model.stim_chosen = choice

                outcome = outcome_value[s, b, p, t]
                confidence = confidence_value[s, b, p, t]

                new_value_choice = model.update(outcome, confidence)
                correct_choice = correct_value[s, b, p, t]

                if correct_choice == False:
                    performance[b, p, t] = 0

                elif correct_choice == True:
                    performance[b, p, t] = 1

                for k in range(nbandits):

                    if k == choice:
                        new_values_choice[b, p, t, k] = new_value_choice
                    else:
                        new_values_choice[b, p, t, k] = 0 if t == 0 else new_values_choice[b, p, t - 1, k]

        true_values_choice[b, :] = true_value[s, b, :]

    if return_full:
        return new_values_choice, true_values_choice, performance
    else:
        return choice_probab[~np.isnan(choice_probab)]


if __name__ == '__main__':

    for m, models in enumerate(modellist):
        for n in range(nsubjects):
            if models != modellist[set_model]:
                continue

            fitting.set_model(n, nsubjects, modellist[m], run_model, nparams[m])
            params = paramlist[m]
            fitting.local_minima(expect[m], bounds[m])

            print(n)

            for param in range(nparams[m]):
                paramfit[n, m, param] = fitting.data[n, param]

            for values in range(len(fitting.choice_probab)):
                probab_choice[n, m, values] = fitting.choice_probab[values]

            negll[n, m] = fitting.negll
            AIC[n, m], BIC[n, m] = fitting.model_fit(negll[n, m])

            locals()["parameter_m" + str(m)] = pd.DataFrame(data={"ALPHA": paramfit[:, m, 0], "BETA": paramfit[:, m, 1],
                                                                  "ALPHA_C": paramfit[:, m, 2], "GAMMA": paramfit[:, m, 3]},
                                                            columns=["ALPHA", "BETA", "ALPHA_C", "GAMMA"])
            locals()["fit_m" + str(m)] = pd.DataFrame(data={"AIC": AIC[:, m], "BIC": BIC[:, m], "NEGLL": negll[:, m]},
                                                      columns=["AIC", "BIC", "NEGLL"])
            locals()["choice_probab_m" + str(m)] = pd.DataFrame(data={"choice_probab_subj" + str(n): probab_choice[n, m, :]},
                                                                columns=["choice_probab_subj" + str(n)])
            saveChoiceProbab = pd.concat([saveChoiceProbab, eval("choice_probab_m" + str(m))], axis=1)
            locals()["param_corr_m" + str(m)] = eval("parameter_m" + str(m)).corr()

# pd.concat([eval("parameter_m" + str(set_model)), eval("fit_m" + str(set_model))], axis=1).to_pickle("./fittingDataM" + str(set_model) + ".pkl")
# saveChoiceProbab.to_pickle("./choiceProbabM" + str(set_model) + ".pkl")
