import os
import numpy as np
import pandas as pd

from ConfLearning.models.rl_simple import Rescorla, RescorlaZero, RescorlaBetaSlope, RescorlaPerservation
from ConfLearning.models.rl_simple_simchoice import RescorlaConf, RescorlaConfGen, RescorlaConfBase, RescorlaConfBaseGen, RescorlaConfZero, RescorlaConfZeroGen, RescorlaConfBaseZero, RescorlaConfBaseZeroGen
from ConfLearning.models.maximum_likelihood import ParameterFit
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

# set_model = 4   # CHANGE HERE

nsubjects = max(matrix.subject.values) + 1
nblocks = max(matrix.block.values) + 1
nphases = max(matrix[~matrix.phase.isna()].phase.values) + 1
# ntrials = 56
ntrials_phase_max = 18
nbandits = 5

alpha = 'ALPHA'
beta = 'BETA'
beta_slope = 'BETA_SLOPE'
eta = 'ETA'
alpha_n = 'ALPHA_N'
alpha_c = 'ALPHA_C'
gamma = 'GAMMA'

lower_alpha, la = 0, 0
lower_beta, lb = 0, 0
lower_beta_slope, lbs = -1, -1
lower_eta, le = -5, -5
lower_alpha_n, lan = 0, 0
lower_alpha_c, lac = 0, 0
lower_gamma, lg = 0, 0

upper_alpha, ua = 1, 1
upper_beta, ub = 2, 2
upper_beta_slope, ubs = 1, 1
upper_eta, ue = 5, 5
upper_alpha_n, uan = 1, 1
upper_alpha_c, uac = 1, 1
upper_gamma, ug = 100, 100

grid_alpha = np.arange(0.1, 0.51, 0.2)
grid_beta = np.arange(0.1, 0.31, 0.1)
grid_beta_slope = np.arange(-0.4, 0.11, 0.1)
grid_eta = np.arange(-0.5, 0.51, 0.1)
grid_alpha_n = np.arange(0.01, 0.061, 0.05)
grid_alpha_c = np.hstack((0, np.arange(0.05, 0.5, 0.05)))      # np.arange(0.05, 0.5, 0.1)
grid_gamma = np.hstack((0, np.arange(0.05, 0.5, 0.05)))    # np.arange(0.05, 0.5, 0.1)


bounds = [np.c_[np.array([la, lb]), np.array([ua, ub])],
          np.c_[np.array([la, lb, lan]), np.array([ua, ub, uan])],
          *[np.c_[np.array([la, lb, lac, lg]), np.array([ua, ub, uac, ug])] for _ in range(4)],
          *[np.c_[np.array([la, lb, lac, lg, lan]), np.array([ua, ub, uac, ug, uan])] for _ in range(4)],
          np.c_[np.array([la, lb, lbs]), np.array([ua, ub, ubs])],
          np.c_[np.array([la, lb, le]), np.array([ua, ub, ue])]
]

expect = [(np.array([ua, ub]) - np.array([la, lb])) / 2,
          (np.array([ua, ub, uan]) - np.array([la, lb, lan])) / 2,
          *[(np.array([ua, ub, uac, ug]) - np.array([la, lb, lac, lg])) / 2 for _ in range(4)],
          *[(np.array([ua, ub, uac, ug, uan]) - np.array([la, lb, lac, lg, lan])) / 2 for _ in range(4)],
          (np.array([ua, ub, ubs]) - np.array([la, lb, lbs])) / 2,
          (np.array([ua, ub, ue]) - np.array([la, lb, le])) / 2
]

grid_range = [
    [grid_alpha, grid_beta],
    [grid_alpha, grid_beta, grid_alpha_n],
    *[[grid_alpha, grid_beta, grid_gamma, grid_alpha_c] for _ in range(4)],
    *[[grid_alpha, grid_beta, grid_gamma, grid_alpha_c, grid_alpha_n] for _ in range(4)],
    [grid_alpha, grid_beta, grid_beta_slope],
    [grid_alpha, grid_beta, grid_eta]
]

# expect = [[0.1, 1], [0.1, 1, 0], [0.1, 1, 0, 0], [0.1, 1, 0, 0], [0.1, 1, 0, 0], [0.1, 1, 0, 0], [0.1, 1, 0, 0]]

modellist = [Rescorla, RescorlaZero, RescorlaConf, RescorlaConfGen, RescorlaConfBase, RescorlaConfBaseGen, RescorlaConfZero, RescorlaConfZeroGen, RescorlaConfBaseZero, RescorlaConfBaseZeroGen, RescorlaBetaSlope, RescorlaPerservation]
model_names = ['Static', 'Deval', 'DualSpec', 'DualUnspec', 'MonoSpec', 'MonoUnspec', 'DualSpecDeval', 'DualUnspecDeval', 'MonoSpecDeval', 'MonoUnspecDeval', 'BetaSlope', 'Perservation']
paramlist = [
    [alpha, beta],
    [alpha, beta, alpha_n],
    *[[alpha, beta, alpha_c, gamma] for _ in range(4)],
    *[[alpha, beta, alpha_c, gamma, alpha_n] for _ in range(4)],
    [alpha, beta, beta_slope],
    [alpha, beta, eta]
]
nparams = [2, 3, 4, 4, 4, 4, 5, 5, 5, 5, 3, 3]
nmodels = len(modellist)

probab_choice = np.full((nmodels, nsubjects, nblocks, nphases, ntrials_phase_max), np.nan)
paramfit = np.full((nmodels, nsubjects, max(nparams)), np.nan)
negll = np.full((nmodels, nsubjects), np.nan)
AIC, BIC = np.full((nmodels, nsubjects), np.nan, float), np.full((nmodels, nsubjects), np.nan)

fittingParams, fittingModel, saveChoiceProbab = None, None, None


def run_model(params, modelspec, s, return_cp=False, return_full=False, return_conf_esti=False, return_nll=False, return_bias=False):

    model = modelspec(*params)

    negLogL = 0

    if return_nll:
        negLoLi = np.full((nblocks, nphases, ntrials_phase_max), np.nan, float)

    if return_cp:
        choiceprob = np.full((nblocks, nphases, ntrials_phase_max), np.nan)

    if return_bias:
        design_bias = np.full((nblocks, nphases, ntrials_phase_max, nbandits), np.nan, float)

    if return_full:
        new_values_choice = np.full((nblocks, nphases, ntrials_phase_max, nbandits), np.nan, float)
        true_values_choice = np.full((nblocks, nbandits), np.nan, float)
        performance = np.full((nblocks, nphases, ntrials_phase_max), np.nan, float)

    if return_conf_esti:
        conf_PE = np.full((nblocks, nphases, ntrials_phase_max, nbandits), np.nan, float)
        conf_expect_pre = np.full((nblocks, nphases, ntrials_phase_max, nbandits), np.nan, float)
        conf_expect_post = np.full((nblocks, nphases, ntrials_phase_max, nbandits), np.nan, float)
        behav_confidence = np.full((nblocks, nphases, ntrials_phase_max, nbandits), np.nan, float)
        model_confidence = np.full((nblocks, nphases, ntrials_phase_max, nbandits), np.nan, float)
        model_confidence2 = np.full((nblocks, nphases, ntrials_phase_max, nbandits), np.nan, float)
        model_perf = np.full((nblocks, nphases, ntrials_phase_max, nbandits), np.nan, float)
        behav_perf = np.full((nblocks, nphases, ntrials_phase_max, nbandits), np.nan, float)

    for b in range(nblocks):

        model.values = np.full(nbandits, 0, float)
        if str(RescorlaBetaSlope) == str(modelspec):
            model.beta = params[1]

        for p in range(nphases):
            for i, t in enumerate(np.where(~np.isnan(stim_left[s, b, p]))[0]):

                model.get_current_trial(t)

                model.stims = np.array([int(stim_left[s, b, p, t]), int(stim_right[s, b, p, t])])
                model.stim_chosen = int(chosen_stim[s, b, p, t])

                cp = model.get_choice_probab(outcome_value[s, b, p, t])
                model.predicted_choice()

                if return_cp:
                    choiceprob[b, p, i] = cp

                negLogL -= np.log(np.maximum(cp, 1e-8))

                if return_nll:
                    negLoLi[b, p, i] = negLogL

                # confidence = 2 * cp - 1 if cp >= 0.5 else 2 * (1 - cp) - 1
                # new_value_choice = model.update(outcome_value[s, b, p, t], confidence_value[s, b, p, t])


                confidence = confidence_value[s, b, p, t] / 10
                # confidence = model.get_confidence2()
                # confidence = model.get_confidence_inv()
                if return_bias:

                    for k in range(nbandits):

                        if k in model.stims:
                            confpe, design_bias[b, p, i, k], confexp = model.get_confidence_exp_pe(confidence)
                        else:
                            if (p > 0) & (i == 0):

                                if np.all(np.isnan(design_bias[b, p - 1, :, k])):
                                    design_bias[b, p, i, k] = design_bias[b, p - 2, :, k][~np.isnan(design_bias[b, p - 2, :, k])][-1]
                                else:
                                    design_bias[b, p, i, k] = design_bias[b, p - 1, :, k][~np.isnan(design_bias[b, p - 1, :, k])][-1]
                            else:
                                design_bias[b, p, i, k] = 0 if t == 0 else design_bias[b, p, i - 1, k]

                if return_conf_esti:
                    for k in range(nbandits):

                        if k == model.stim_chosen:
                            if hasattr(model, 'get_confidence_exp_pe'):
                                conf_PE[b, p, i, k], conf_expect_pre[b, p, i, k], conf_expect_post[b, p, i, k] = model.get_confidence_exp_pe(confidence)
                            behav_confidence[b, p, i, k] = confidence
                            model_confidence[b, p, i, k] = model.get_choice_probab(outcome_value[s, b, p, t])
                            model_confidence2[b, p, i, k] = model.get_confidence()
                            if true_value[s, b, int(stim_right[s, b, p, t])] >= true_value[s, b, int(stim_left[s, b, p, t])]:
                                model_perf[b, p, i, k] = model.get_choice_probab2()
                            else:
                                model_perf[b, p, i, k] = 1 - model.get_choice_probab2()
                            # print(model_perf[b, p, i, k])
                            stim_chosen = int(chosen_stim[s, b, p, t])
                            stim_unchosen = int(stim_right[s, b, p, t]) if chosen_stim[s, b, p, t] == stim_left[s, b, p, t] else int(stim_left[s, b, p, t])
                            behav_perf[b, p, i, k] = int(true_value[s, b, stim_chosen] >= true_value[s, b, stim_unchosen])


                        else:
                            if (p > 0) & (i == 0):

                                if hasattr(model, 'get_confidence_exp_pe'):
                                    if np.all(np.isnan(conf_PE[b, p - 1, :, k])):
                                        conf_PE[b, p, i, k] = conf_PE[b, p - 2, :, k][~np.isnan(conf_PE[b, p - 2, :, k])][-1]
                                    else:
                                        conf_PE[b, p, i, k] = conf_PE[b, p - 1, :, k][~np.isnan(conf_PE[b, p - 1, :, k])][-1]

                                    if np.all(np.isnan(conf_expect_pre[b, p - 1, :, k])):
                                        conf_expect_pre[b, p, i, k] = conf_expect_pre[b, p - 2, :, k][~np.isnan(conf_expect_pre[b, p - 2, :, k])][-1]
                                    else:
                                        conf_expect_pre[b, p, i, k] = conf_expect_pre[b, p - 1, :, k][~np.isnan(conf_expect_pre[b, p - 1, :, k])][-1]

                                if np.all(np.isnan(behav_confidence[b, p - 1, :, k])):
                                    behav_confidence[b, p, i, k] = behav_confidence[b, p - 2, :, k][~np.isnan(behav_confidence[b, p - 2, :, k])][-1]
                                    model_confidence[b, p, i, k] = model_confidence[b, p - 2, :, k][~np.isnan(model_confidence[b, p - 2, :, k])][-1]
                                    model_confidence2[b, p, i, k] = model_confidence2[b, p - 2, :, k][~np.isnan(model_confidence2[b, p - 2, :, k])][-1]
                                    model_perf[b, p, i, k] = model_perf[b, p - 2, :, k][~np.isnan(model_perf[b, p - 2, :, k])][-1]
                                    behav_perf[b, p, i, k] = behav_perf[b, p - 2, :, k][~np.isnan(behav_perf[b, p - 2, :, k])][-1]
                                else:
                                    behav_confidence[b, p, i, k] = behav_confidence[b, p - 1, :, k][~np.isnan(behav_confidence[b, p - 1, :, k])][-1]
                                    model_confidence[b, p, i, k] = model_confidence[b, p - 1, :, k][~np.isnan(model_confidence[b, p - 1, :, k])][-1]
                                    model_confidence2[b, p, i, k] = model_confidence2[b, p - 1, :, k][~np.isnan(model_confidence2[b, p - 1, :, k])][-1]
                                    model_perf[b, p, i, k] = model_perf[b, p - 1, :, k][~np.isnan(model_perf[b, p - 1, :, k])][-1]
                                    behav_perf[b, p, i, k] = behav_perf[b, p - 1, :, k][~np.isnan(behav_perf[b, p - 1, :, k])][-1]

                            else:
                                if hasattr(model, 'get_confidence_exp_pe'):
                                    conf_PE[b, p, i, k] = 0 if t == 0 else conf_PE[b, p, i - 1, k]
                                    conf_expect_pre[b, p, i, k] = 0 if t == 0 else conf_expect_pre[b, p, i - 1, k]
                                behav_confidence[b, p, i, k] = 0 if t == 0 else behav_confidence[b, p, i - 1, k]
                                model_confidence[b, p, i, k] = 0 if t == 0 else model_confidence[b, p, i - 1, k]
                                model_confidence2[b, p, i, k] = 0 if t == 0 else model_confidence2[b, p, i - 1, k]
                                model_perf[b, p, i, k] = 0.5 if t == 0 else model_perf[b, p, i - 1, k]
                                behav_perf[b, p, i, k] = 0.5 if t == 0 else behav_perf[b, p, i - 1, k]

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

    if return_conf_esti:
        return conf_PE, conf_expect_pre, behav_confidence, model_confidence, model_confidence2, model_perf, behav_perf

    if return_nll:
        return negLoLi

    if return_bias:
        return design_bias

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
                continue
            print(f'[Model {m+1} / {len(modellist)}] Fitting model {modellist[set_model]}')

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

            # locals()["parameter_m" + str(m)] = pd.DataFrame(data={"ALPHA": paramfit[m, :, 0], "BETA": paramfit[m, :, 1],
            #                                                       "ALPHA_C": paramfit[m, :, 2], "GAMMA": paramfit[m, :, 3], "ALPHA_N": paramfit[m, :, 4]},
            #                                                 columns=["ALPHA", "BETA", "ALPHA_C", "GAMMA", "ALPHA_N"])
            locals()["parameter_m" + str(m)] = pd.DataFrame(data={params[i]: paramfit[m, :, i] for i in range(nparams[m])}, columns=params)
            locals()["fit_m" + str(m)] = pd.DataFrame(data={"AIC": AIC[m, :], "BIC": BIC[m, :], "NEGLL": negll[m, :]},
                                                      columns=["AIC", "BIC", "NEGLL"])
            locals()["choice_probab_m" + str(m)] = pd.DataFrame(data={"choice_probab_subj" + str(n): probab_choice[m, n][~np.isnan(probab_choice[m, n])]},
                                                                columns=["choice_probab_subj" + str(n)])
            saveChoiceProbab = pd.concat([saveChoiceProbab, eval("choice_probab_m" + str(m))], axis=1)
            locals()["param_corr_m" + str(m)] = eval("parameter_m" + str(m)).corr()

            # pd.concat([eval("parameter_m" + str(m)), eval("fit_m" + str(m))], axis=1).to_pickle("../results/fittingData/fittingDataM" + str(m) + "_cp.pkl", protocol=4)
            pd.concat([eval("parameter_m" + str(m)), eval("fit_m" + str(m))], axis=1).to_pickle(f"../results/fittingData/fittingData_{model_names[m]}_simchoice.pkl", protocol=4)
        # saveChoiceProbab.to_pickle("../results/choiceProbab/choiceProbabM" + str(m) + "_cp.pkl", protocol=4)
