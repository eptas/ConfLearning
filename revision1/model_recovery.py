import os
from typing import Union

import numpy as np
import pandas as pd

from pathlib import Path
from ConfLearning.models.rl_simple import Rescorla, RescorlaZero as RescorlaDeval, RescorlaPerseveration
from ConfLearning.models.rl_simple_choice_simchoice import RescorlaChoiceMono
from ConfLearning.models.rl_simple_simchoice import RescorlaConfBase, RescorlaConfBaseGen
from ConfLearning.models.maximum_likelihood import ParameterFit
from ConfLearning.recovery.bandit import BanditMoney
from ConfLearning.recovery.design import Design

# initialize data simulation

sim_data = False

cwd = Path.cwd()

path_data = os.path.join(cwd, '../data/')
matrix = pd.read_pickle(os.path.join(path_data, 'data.pkl'))

nsubjects = max(matrix.subject.values) + 1
nblocks = max(matrix.block.values) + 1
nphases = max(matrix[~matrix.phase.isna()].phase.values) + 1

ntrials_phase_max = 18
nbandits = 5

modellist = [Rescorla, RescorlaDeval, RescorlaChoiceMono, RescorlaConfBase, RescorlaConfBaseGen, RescorlaPerseveration]
model_names = ['Static', 'Deval', 'Choice', 'ConfSpec', 'ConfUnspec', 'Perservation']
model_file_names = ['Static', 'Deval', 'Mono_choice', 'MonoSpec', 'MonoUnspec', 'Perservation']
paramlist = [['ALPHA', 'BETA'], ['ALPHA', 'BETA', 'ALPHA_C'], ['ALPHA', 'BETA', 'GAMMA'], *[['ALPHA', 'BETA', 'ALPHA_C', 'GAMMA'] for _ in range(2)], ['ALPHA', 'BETA', 'ETA']]   # for behav fitting dataframe

var_list = ['stim_left', 'stim_right', 'history_constraint']
stim_left, stim_right, history_constraint = None, None, None

for v, variable in enumerate(var_list):
    locals()[variable] = np.load(os.path.join(path_data, variable + '.npy'))


n_datasets = nsubjects

choice = np.full((len(modellist), nsubjects, nblocks, nphases, ntrials_phase_max), np.nan)
out_val = np.full((len(modellist), nsubjects, nblocks, nphases, ntrials_phase_max), np.nan)
conf_val = np.full((len(modellist), nsubjects, nblocks, nphases, ntrials_phase_max), np.nan)

alpha = 'ALPHA'
beta = 'BETA'
beta_slope = 'BETA_SLOPE'
eta = 'ETA'
alpha_n = 'ALPHA_N'
alpha_c = 'ALPHA_C'
gamma = 'GAMMA'

la, ua = 0, 1
lb, ub = 0, 2
lan, uan = 0, 1
lg, ug = 0, 100
le, ue = -5, 5

grid_alpha = np.arange(0.1, 0.51, 0.2)
grid_beta = np.arange(0.1, 0.31, 0.1)
grid_alpha_n = np.arange(0.01, 0.061, 0.05)
grid_gamma = np.hstack((0, np.arange(0.05, 0.5, 0.05)))    # np.arange(0.05, 0.5, 0.1)
grid_eta = np.arange(-0.5, 0.51, 0.1)

bounds = [np.c_[np.array([la, lb]), np.array([ua, ub])],
          np.c_[np.array([la, lb, lan]), np.array([ua, ub, uan])],
          np.c_[np.array([la, lb, lg]), np.array([ua, ub, ug])],
          *[np.c_[np.array([la, lb, lan, lg]), np.array([ua, ub, uan, ug])] for _ in range(2)],
          np.c_[np.array([la, lb, le]), np.array([ua, ub, ue])]
]

expect = [(np.array([ua, ub]) - np.array([la, lb])) / 2,
          (np.array([ua, ub, uan]) - np.array([la, lb, lan])) / 2,
          (np.array([ua, ub, ug]) - np.array([la, lb, lg])) / 2,
          *[(np.array([ua, ub, uan, ug]) - np.array([la, lb, lan, lg])) / 2 for _ in range(2)],
          (np.array([ua, ub, ue]) - np.array([la, lb, le])) / 2
]

grid_range = [
    [grid_alpha, grid_beta],
    [grid_alpha, grid_beta, grid_alpha_n],
    [grid_alpha, grid_beta, grid_gamma],
    *[[grid_alpha, grid_beta, grid_alpha_n, grid_gamma] for _ in range(2)],
    [grid_alpha, grid_beta, grid_eta]
]

param_list = [[alpha, beta], [alpha, beta, alpha_n], [alpha, beta, gamma], *[[alpha, beta, alpha_n, gamma] for _ in range(2)], [alpha, beta, eta]]
nparams = [2, 3, 3, 4, 4, 3]

fitting = ParameterFit()

probab_choice = np.full((len(modellist), len(modellist), nsubjects, nblocks, nphases, ntrials_phase_max), np.nan)
saveParameters, saveFitting, saveChoiceProbab = None, None, None

paramfit = np.full((len(modellist), len(modellist), nsubjects, max(nparams)), np.nan)
negll = np.full((len(modellist), len(modellist), nsubjects), np.nan)
AIC, BIC = np.full((len(modellist), len(modellist), nsubjects), np.nan, float), np.full((len(modellist), len(modellist), nsubjects), np.nan)

# simulate data

if sim_data == True:

    for m, model in enumerate(modellist):

        param = paramlist[m]
        fit_data = pd.read_pickle(os.path.join(path_data, "../results/fittingData/fittingData_" + model_file_names[m] + "_simchoice.pkl"))

        for i in range(n_datasets):  # corresponds to subjects

            print(f'[{model_names[m]} {m + 1} / {len(modellist)}] Subject {i + 1} / {n_datasets}')

            parameter = []

            for para in range(len(param)):
                parameter = np.append(parameter, fit_data.eval(param[para])[i])

            sim_model = model(*parameter)

            np.random.seed(i)

            bandit = BanditMoney()
            design = Design()

            for b in range(nblocks):

                sim_model.values = np.full(nbandits, 0, float)

                bandit.reset_outcome_history()
                bandit.set_outcome_schedule(design.outcome_schedule[b], design.outcome_base[b], design.outcome_diff[b])

                for p in range(nphases):
                    for t, tri in enumerate(np.where(~np.isnan(stim_left[i, b, p]))[0]):

                        sim_model.noise = np.random.rand(1)

                        sim_model.get_current_trial(tri)
                        sim_model.stims = np.array([int(stim_left[i, b, p, tri]), int(stim_right[i, b, p, tri])])

                        cp = sim_model.get_choice_probab(out_val[m, i, b, p, t])
                        choice[m, i, b, p, t], choice_index = sim_model.simulated_choice()

                        sim_model.stim_chosen = int(choice[m, i, b, p, t])

                        out = bandit.sample(int(choice[m, i, b, p, t]), ignore_history_constraints=history_constraint[i, b, p, tri])
                        out_val[m, i, b, p, t] = out if (p != 1) else np.nan

                        conf_val[m, i, b, p, t] = sim_model.simulated_confidence(choice_index)

                        sim_model.update(out_val[m, i, b, p, t], conf_val[m, i, b, p, t])

sim_variables = ['choice', 'out_val', 'conf_val']

simu_data = ['choice_value', 'out_value', 'conf_value']
choice_value, out_value, conf_value = None, None, None

for v, var in enumerate(sim_variables):

    if (sim_data == True):
        np.save(var, eval(var))
    else:
        locals()[simu_data[v]] = np.load(var + '.npy')


def fit_model(parame, running_model, s, simulation_model, return_cp=False):

    modelling = running_model(*parame)
    sim_index = modellist.index(simulation_model)

    negLogL = 0
    negLoLi = np.full((nblocks, nphases, ntrials_phase_max), np.nan, float)

    if return_cp:
        choiceprob = np.full((nblocks, nphases, ntrials_phase_max), np.nan)

    for b in range(nblocks):

        modelling.values = np.full(nbandits, 0, float)

        for p in range(nphases):
            for t, tri in enumerate(np.where(~np.isnan(stim_left[s, b, p]))[0]):

                modelling.get_current_trial(tri)

                modelling.stims = np.array([int(stim_left[s, b, p, tri]), int(stim_right[s, b, p, tri])])
                modelling.stim_chosen = int(choice_value[sim_index, s, b, p, t])
                # try:
                #     modelling.stim_chosen = int(choice_value[sim_index, s, b, p, t])
                # except:
                #     a = 3

                cp = modelling.get_choice_probab(out_value[sim_index, s, b, p, t])

                if return_cp:
                    choiceprob[b, p, t] = cp

                negLogL -= np.log(np.maximum(cp, 1e-8))
                negLoLi[b, p, t] = negLogL

                modelling.update(out_value[sim_index, s, b, p, t], conf_value[sim_index, s, b, p, t])

    return (negLogL, choiceprob) if (return_cp == True) else negLogL


if __name__ == '__main__':

    for run, run_mode in enumerate(modellist):
        for simu, simu_model in enumerate(modellist):

            print(f'[run {run + 1} / {len(modellist)}] simu {simu + 1} / {len(modellist)}')

            # if (simu == 0) & (run == 5):
            if True:
                fit_data = pd.read_pickle(os.path.join(path_data, "../results/fittingData/fittingData_" + model_file_names[simu] + "_simchoice.pkl"))
                for n in range(nsubjects):

                    print(f"True params: [{', '.join([f'{v}={fit_data.loc[fit_data.index == n, v].item():.3f}' for v in paramlist[simu]])}] [{', '.join([f'{fit_data.loc[fit_data.index == n, v].item():.3f}' for v in paramlist[simu]])}]")

                    params = param_list[run]
                    fitting.set_model(n, nsubjects, run_mode, fit_model, nparams[run], simu_model)
                    fitting.local_minima(expect[run], bounds[run], grid_range[run])

                    for pa in range(nparams[run]):
                        paramfit[run, simu, n, pa] = min(bounds[run][pa][1], max(bounds[run][pa][0], fitting.data[n, pa]))

                    negll[run, simu, n], probab_choice[run, simu, n] = fit_model(fitting.data[n], run_mode, n, simu_model, return_cp=True)
                    nsamples = np.sum(~np.isnan(probab_choice[run, simu, n]))

                    AIC[run, simu, n], BIC[run, simu, n] = fitting.model_fit(negll[run, simu, n], nsamples)

                    if n == (nsubjects - 1):
                        parameter_fit = pd.DataFrame(data={f'{params[i]}_{run}_{simu}e': paramfit[run, simu, :, i] for i in range(nparams[run])})
                                                     # columns=["ALPHA_" + str(run) + '_' + str(simu) + 'e',
                                                     #          "BETA_" + str(run) + '_' + str(simu) + 'e',
                                                     #          "ALPH_N_" + str(run) + '_' + str(simu) + 'e',
                                                     #          "GAMMA_" + str(run) + '_' + str(simu) + 'e'
                                                     #          ])

                        saveParameters = pd.concat([saveParameters, parameter_fit], axis=1)

                        model_fit = pd.DataFrame(data={"AIC_" + str(run) + '_' + str(simu) + 'e': AIC[run, simu, :],
                                                       "BIC_" + str(run) + '_' + str(simu) + 'e': BIC[run, simu, :],
                                                       "NEGLL_" + str(run) + '_' + str(simu) + 'e': negll[run, simu, :]
                                                       },
                                                 columns=["AIC_" + str(run) + '_' + str(simu) + 'e',
                                                          "BIC_" + str(run) + '_' + str(simu) + 'e',
                                                          "NEGLL_" + str(run) + '_' + str(simu) + 'e'
                                                          ])

                        saveFitting = pd.concat([saveFitting, model_fit], axis=1)

                        choice_probability = pd.DataFrame(data={"cp_" + str(run) + '_' + str(simu) + 'e': probab_choice[run, simu, :][~np.isnan(probab_choice[run, simu, :])]},
                                                          columns=["cp_" + str(run) + '_' + str(simu) + 'e'])

                        saveChoiceProbab = pd.concat([saveChoiceProbab, choice_probability], axis=1)

        if run == (len(modellist) - 1):

            pd.concat([saveParameters, saveFitting], axis=1).to_pickle("fittingData_" + model_names[run] + ".pkl", protocol=4)
            saveChoiceProbab.to_pickle("choiceProbabM" + str(run) + ".pkl", protocol=4)
