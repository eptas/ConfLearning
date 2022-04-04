import os
import numpy as np
import pandas as pd

from pathlib import Path
from itertools import product as prod

from ConfLearning.models.rl_simple import Rescorla, RescorlaZero as RescorlaDeval, RescorlaPerservation
from ConfLearning.models.rl_simple_choice_simchoice import RescorlaChoiceMono
from ConfLearning.models.rl_simple_simchoice import RescorlaConfBase, RescorlaConfBaseGen

from ConfLearning.models.maximum_likelihood import ParameterFit
from ConfLearning.recovery.bandit import BanditMoney
from ConfLearning.recovery.gen_design import GenDesign

# initialize data simulation

cwd = Path.cwd()
fitting = ParameterFit()

nsubjects = 100
n_datasets = nsubjects

nblocks = 11
nphases = 3
ntrials_phase_max = 57
nbandits = 5

modellist = [Rescorla, RescorlaDeval, RescorlaChoiceMono, RescorlaConfBase, RescorlaConfBaseGen, RescorlaPerservation]
model_names = ['Static', 'Deval', 'Choice', 'ConfSpec', 'ConfUnspec', 'Perservation']
paramlist = [['alpha', 'beta'], ['alpha', 'beta', 'alpha_c'], ['alpha', 'beta', 'gamma'], *[['alpha', 'beta', 'alpha_c', 'gamma'] for _ in range(2)], ['alpha', 'beta', 'eta']]   # for behav fitting dataframe

var_list = ['stim_left', 'stim_right', 'history_constraint']
stim_left, stim_right, history_constraint = None, None, None

for v, variable in enumerate(var_list):
    locals()[variable] = np.load(os.path.join(cwd, variable + '_10.npy'))

alpha = np.arange(0, 1.00001, 0.25)
beta = np.arange(0.02, 2.00001, step=((2 - 0.02) / 4))                # alternatively 0.03 ?
alpha_c = np.arange(0, 1.00001, 0.25)
gamma = np.arange(0, 10.00001, 2.5)
eta = np.arange(-1.5, 1.500001, 0.75)

choice = np.full((n_datasets, nblocks, nphases, ntrials_phase_max), np.nan)
out_val = np.full((n_datasets, nblocks, nphases, ntrials_phase_max), np.nan)
conf_val = np.full((n_datasets, nblocks, nphases, ntrials_phase_max), np.nan)

la, ua = 0, 1
lb, ub = 0, 2
lac, uac = 0, 1
lg, ug = 0, 100
le, ue = -5, 5

grid_alpha = np.arange(0.1, 0.51, 0.2)
grid_beta = np.arange(0.1, 0.31, 0.1)
grid_alpha_c = np.arange(0.01, 0.061, 0.05)
grid_gamma = np.hstack((0, np.arange(0.05, 0.5, 0.05)))    # np.arange(0.05, 0.5, 0.1)
grid_eta = np.arange(-0.5, 0.51, 0.1)

bounds = [np.c_[np.array([la, lb]), np.array([ua, ub])],
          np.c_[np.array([la, lb, lac]), np.array([ua, ub, uac])],
          np.c_[np.array([la, lb, lg]), np.array([ua, ub, ug])],
          *[np.c_[np.array([la, lb, lac, lg]), np.array([ua, ub, uac, ug])] for _ in range(2)],
          np.c_[np.array([la, lb, le]), np.array([ua, ub, ue])]
]

expect = [(np.array([ua, ub]) - np.array([la, lb])) / 2,
          (np.array([ua, ub, uac]) - np.array([la, lb, lac])) / 2,
          (np.array([ua, ub, ug]) - np.array([la, lb, lg])) / 2,
          *[(np.array([ua, ub, uac, ug]) - np.array([la, lb, lac, lg])) / 2 for _ in range(2)],
          (np.array([ua, ub, ue]) - np.array([la, lb, le])) / 2
]

grid_range = [
    [grid_alpha, grid_beta],
    [grid_alpha, grid_beta, grid_alpha_c],
    [grid_alpha, grid_beta, grid_gamma],
    *[[grid_alpha, grid_beta, grid_alpha_c, grid_gamma] for _ in range(2)],
    [grid_alpha, grid_beta, grid_eta]
]

nparams = [2, 3, 3, 4, 4, 3]

probab_choice = np.full((len(modellist), len(modellist), len(alpha)**max(nparams), nsubjects, n_datasets, nblocks, nphases, ntrials_phase_max), np.nan)
negll = np.full((len(modellist), len(modellist), len(alpha)**max(nparams), nsubjects), np.nan)
AIC, BIC = np.full((len(modellist), len(modellist), len(alpha)**max(nparams), nsubjects), np.nan, float), np.full((len(modellist), len(modellist), nsubjects), np.nan)


def sim_data(subj, simulation_model, simulation_parameter):

    parameter = [eval(x) for x in simulation_parameter]
    simModel = modellist[eval(simulation_model)](*parameter)

    np.random.seed(subj)
    simModel.noise = np.random.rand(1)

    for i in np.arange(0, n_datasets):

        print("Simulating dataset " + str(i) + " for subj " + str(subj))

        bandit = BanditMoney()
        design = GenDesign()

        for b in range(nblocks):

            simModel.values = np.full(nbandits, 0, float)

            bandit.reset_outcome_history()
            bandit.set_outcome_schedule(design.outcome_schedule[b], design.outcome_base[b], design.outcome_diff[b])

            for p in range(nphases):
                for t, tri in enumerate(np.where(~np.isnan(stim_left[i, b, p]))[0]):

                    simModel.get_current_trial(tri)
                    simModel.stims = np.array([int(stim_left[i, b, p, tri]), int(stim_right[i, b, p, tri])])

                    cp = simModel.get_choice_probab()

                    choice[i, b, p, t], choice_index = simModel.simulated_choice()
                    simModel.stim_chosen = int(choice[i, b, p, t])

                    out = bandit.sample(int(choice[i, b, p, t]), ignore_history_constraints=history_constraint[i, b, p, tri])
                    out_val[i, b, p, t] = out if (p != 1) else np.nan

                    conf_val[i, b, p, t] = simModel.simulated_confidence(choice_index)

                    simModel.update(out_val[i, b, p, t], conf_val[i, b, p, t])

    return choice, out_val, conf_val


def fit_model(parame, running_model, s, simulation_model, return_cp=False):

    sim_model, sim_paras = simulation_model[-1], simulation_model[:-1]
    choices, outcome_value, confidence_value = sim_data(s, sim_model, sim_paras)

    modelling = running_model(*parame)

    negLogL = 0

    if return_cp:
        choiceprob = np.full((n_datasets, nblocks, nphases, ntrials_phase_max), np.nan)

    for i in np.arange(0, n_datasets):
        for b in range(nblocks):

            modelling.values = np.full(nbandits, 0, float)

            for p in range(nphases):
                for t, tri in enumerate(np.where(~np.isnan(stim_left[i, b, p]))[0]):

                    modelling.get_current_trial(tri)
                    modelling.stims = np.array([int(stim_left[i, b, p, tri]), int(stim_right[i, b, p, tri])])

                    cp = modelling.get_choice_probab()

                    modelling.stim_chosen = int(choices[i, b, p, t])

                    modelling.update(outcome_value[i, b, p, t], confidence_value[i, b, p, t])

                    if return_cp:
                        choiceprob[i, b, p, t] = cp

                    negLogL -= np.log(np.maximum(cp, 1e-8))

    return (negLogL, choiceprob) if (return_cp == True) else negLogL


if __name__ == '__main__':

    for run, run_mode in enumerate(modellist):

        saveFitting = None

        for simu, simu_model in enumerate(modellist):

            print(f'[run {run + 1} / {len(modellist)}] simu {simu + 1} / {len(modellist)}')

            curr_parameter = [eval(x) for x in paramlist[simu]]

            if nparams[simu] == 2:
                param_combi = list(prod(list(curr_parameter[0]), list(curr_parameter[1])))

            elif nparams[simu] == 3:
                param_combi = list(prod(list(curr_parameter[0]), list(curr_parameter[1]), list(curr_parameter[2])))

            else:
                param_combi = list(prod(list(curr_parameter[0]), list(curr_parameter[1]), list(curr_parameter[2]), list(curr_parameter[3])))

            for pa, para in enumerate(param_combi):

                print("Fitting param combi " + str(pa) + " / " + str(len(param_combi)))

                simu_id = []

                simu_id.append(list(alpha).index(para[0]))
                simu_id.append(list(beta).index(para[1]))

                if simu in [1, 3, 4]:
                    simu_id.append(list(alpha_c).index(para[2]))

                if simu in [2, 3, 4]:
                    simu_id.append(list(gamma).index(para[3])) if simu != 2 else simu_id.append(list(gamma).index(para[2]))

                if simu == 5:
                    simu_id.append(list(eta).index(para[2]))

                simu_id.append(simu)
                [str(x) for x in simu_id]

                for n in range(nsubjects):

                    fitting.set_model(n, nsubjects, run_mode, fit_model, nparams[run], simu_id)
                    fitting.local_minima(expect[run], bounds[run], grid_range[run])

                    for pa in range(nparams[run]):
                        paramfit = min(bounds[run][pa][1], max(bounds[run][pa][0], fitting.data[n, pa]))

                    negll[run, simu, pa, n], probab_choice[run, simu, pa, n] = fit_model(fitting.data[n], run_mode, n, simu_model, return_cp=True)
                    nsamples = np.sum(~np.isnan(probab_choice[run, simu, pa, n]))

                    AIC[run, simu, pa, n], BIC[run, simu, pa, n] = fitting.model_fit(negll[run, simu, pa, n], nsamples)

                    if n == (nsubjects - 1):

                        model_fit = pd.DataFrame(data={"AIC_r" + str(run) + '_s' + str(simu) + '_p' + str(pa): AIC[run, simu, pa, :]
                                                       },
                                                 columns=["AIC_r" + str(run) + '_s' + str(simu) + '_p' + str(pa)
                                                          ])

                        saveFitting = pd.concat([saveFitting, model_fit], axis=1)

        if run == (len(modellist) - 1):
            saveFitting.to_pickle("model_recov_" + model_names[run] + ".pkl", protocol=4)
