#!/fast/home/users/ptasczle_c/python/bin/python3 -u
from concurrent.futures import ProcessPoolExecutor as Pool
from timeit import default_timer
import os
import numpy as np
import pandas as pd

from pathlib import Path
from itertools import product as prod
import sys
sys.path.extend(['/fast/users/ptasczle_c/work/Dropbox/confidence/'])
from ConfLearning.models.rl_simple import Rescorla, RescorlaZero as RescorlaDeval, RescorlaPerservation
from ConfLearning.models.rl_simple_choice_simchoice import RescorlaChoiceMono
from ConfLearning.models.rl_simple_simchoice import RescorlaConfBase, RescorlaConfBaseGen

from ConfLearning.models.maximum_likelihood import ParameterFit
from ConfLearning.recovery.bandit import BanditMoney
from ConfLearning.recovery.gen_design import GenDesign

# initialize data simulation

fitting = ParameterFit()
GenDesign.factor = 1
design = GenDesign()

nsubjects = 100
# nsubjects = 2
n_datasets = nsubjects

nblocks = 11
nphases = 3
ntrials_phase_max = 57
nbandits = 5

modellist = [Rescorla, RescorlaDeval, RescorlaChoiceMono, RescorlaConfBase, RescorlaConfBaseGen, RescorlaPerservation]
# modellist = [Rescorla, RescorlaDeval]
model_names = ['Static', 'Deval', 'Choice', 'ConfSpec', 'ConfUnspec', 'Perservation']
paramlist = [['alpha', 'beta'], ['alpha', 'beta', 'alpha_c'], ['alpha', 'beta', 'gamma'], *[['alpha', 'beta', 'alpha_c', 'gamma'] for _ in range(2)], ['alpha', 'beta', 'eta']]   # for behav fitting dataframe

var_list = ['stim_left', 'stim_right', 'history_constraint']
stim_left, stim_right, history_constraint = None, None, None


try:
    # stim_left, stim_right, history_constraint = [np.load(os.path.join('/fast/users/ptasczle_c/work/Dropbox/confidence/ConfLearning/revisions/', f'{v}_10.npy')) for v in var_list]
    stim_left, stim_right, history_constraint = [np.load(os.path.join('/fast/users/ptasczle_c/work/Dropbox/confidence/ConfLearning/revisions/', f'{v}.npy')) for v in var_list]
except:
    cwd = Path.cwd()
    for v, variable in enumerate(var_list):
        # locals()[variable] = np.load(os.path.join(cwd, variable + '_10.npy'))
        locals()[variable] = np.load(os.path.join(cwd, variable + '.npy'))

alpha = np.linspace(0.1, 1, 5)
beta = np.array([0.1, 0.2, 0.4, 0.8, 1.6])
alpha_c = np.linspace(0.1, 1, 5)
gamma = np.exp(np.linspace(np.log(0.1), np.log(10), 5))
eta = np.linspace(-1.5, 1.5, 6)



# alpha = [0.1, 0.2]
# beta = [0.1, 0.2]
# alpha_c = [0.1, 0.2]
# gamma = [0.1, 0.2]
# eta = [0.1, 0.2]

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

# probab_choice = np.full((len(modellist), len(modellist), len(alpha)**max(nparams), nsubjects, n_datasets, nblocks, nphases, ntrials_phase_max), np.nan)
# negll = np.full((len(modellist), len(modellist), len(alpha)**max(nparams), nsubjects), np.nan)
# AIC, BIC = np.full((len(modellist), len(modellist), len(alpha)**max(nparams), nsubjects), np.nan, float), np.full((len(modellist), len(modellist), nsubjects), np.nan)


def sim_data(simulation_model, simulation_parameter):

    # parameter = [eval(x) for x in simulation_parameter]
    parameter = simulation_parameter
    simModel = modellist[simulation_model](*parameter)

    for i in np.arange(0, n_datasets):

        np.random.seed(i)

        # print(f'\t\tSimulating dataset {i + 1} / {n_datasets}')

        bandit = BanditMoney()
        design.generate()

        for b in range(nblocks):

            simModel.values = np.full(nbandits, 0, float)

            bandit.reset_outcome_history()
            bandit.set_outcome_schedule(design.outcome_schedule[b], design.outcome_base[b], design.outcome_diff[b])

            for p in range(nphases):
                for t, tri in enumerate(np.where(~np.isnan(stim_left[i, b, p]))[0]):
                # for t, tri in enumerate(np.where(~design.design[(design.design.block == b) & (design.design.phase == p)].stimulus_left.isna())[0]):
                #     cond = (design.design.block == b) & (design.design.phase == p) & (design.design.trial_phase == tri)
                    simModel.noise = np.random.rand(1)

                    simModel.get_current_trial(tri)
                    simModel.stims = np.array([int(stim_left[i, b, p, tri]), int(stim_right[i, b, p, tri])])
                    # simModel.stims = np.array([design.design[cond].stimulus_left.item(), design.design[cond].stimulus_right.item()])

                    simModel.get_choice_probab()

                    choice[i, b, p, t], choice_index = simModel.simulated_choice()
                    simModel.stim_chosen = int(choice[i, b, p, t])

                    out = bandit.sample(int(choice[i, b, p, t]), ignore_history_constraints=history_constraint[i, b, p, tri])
                    # out = bandit.sample(int(choice[i, b, p, t]), ignore_history_constraints=design.design[cond].pre_equalshown_secondlasttrial.item())
                    out_val[i, b, p, t] = out if (p != 1) else np.nan

                    conf_val[i, b, p, t] = simModel.simulated_confidence(choice_index)

                    simModel.update(out_val[i, b, p, t], conf_val[i, b, p, t])

    return choice, out_val, conf_val


def fit_model(parame, running_model, s, sim_params, choices, outcome_value, confidence_value, return_cp=False):

    modelling = running_model(*parame)

    negLogL = 0

    if return_cp:
        choiceprob = np.full((nblocks, nphases, ntrials_phase_max), np.nan)

    for b in range(nblocks):

        modelling.values = np.full(nbandits, 0, float)

        for p in range(nphases):
            for t, tri in enumerate(np.where(~np.isnan(stim_left[s, b, p]))[0]):

                modelling.get_current_trial(tri)
                modelling.stims = np.array([int(stim_left[s, b, p, tri]), int(stim_right[s, b, p, tri])])

                cp = modelling.get_choice_probab()

                modelling.stim_chosen = int(choices[s, b, p, t])

                modelling.update(outcome_value[s, b, p, t], confidence_value[s, b, p, t])

                if return_cp:
                    choiceprob[b, p, t] = cp

                negLogL -= np.log(np.maximum(cp, 1e-8))

    return (negLogL, choiceprob) if (return_cp == True) else negLogL

def loop(i):

    t0 = default_timer()

    pa, para = i, param_combi[i]

    # simu_id = []
    #
    # simu_id.append(list(alpha).index(para[0]))
    # simu_id.append(list(beta).index(para[1]))
    #
    # if simu in [1, 3, 4]:
    #     simu_id.append(list(alpha_c).index(para[2]))
    #
    # if simu in [2, 3, 4]:
    #     simu_id.append(list(gamma).index(para[3])) if simu != 2 else simu_id.append(list(gamma).index(para[2]))
    #
    # if simu == 5:
    #     simu_id.append(list(eta).index(para[2]))

    # simu_id.append(simu)
    # [str(x) for x in simu_id]
    # ts = default_timer()
    choices, outcome_value, confidence_value = sim_data(simu, para)
    # print(f'\t\tSimulation took {default_timer() - ts:.1f} secs')
    AIC = np.full(nsubjects, np.nan)

    for n in range(nsubjects):

        t1 = default_timer()

        # print(f'\tSubject {n + 1} / {nsubjects}')

        fitting.set_model(n, nsubjects, run_model, fit_model, nparams[run])
        fitting.local_minima(expect[run], bounds[run], grid_range[run], grid_multiproc=False, verbose=False, args=[choices, outcome_value, confidence_value])

        # for p in range(nparams[run]):
        #     paramfit = min(bounds[run][p][1], max(bounds[run][p][0], fitting.data[n, p]))
        negll, probab_choice = fit_model(fitting.data[n], run_model, n, simu_model, choices, outcome_value, confidence_value, return_cp=True)
        nsamples = np.sum(~np.isnan(probab_choice))

        AIC[n], BIC = fitting.model_fit(negll, nsamples)

        if n == (nsubjects - 1):

            model_fit = pd.DataFrame(data={
                # 'fit_model': model_names[run],
                'subject': range(nsubjects),
                "AIC_r" + str(run) + '_s' + str(simu) + '_p' + str(pa): AIC
                                           },
                                     columns=['subject', "AIC_r" + str(run) + '_s' + str(simu) + '_p' + str(pa)
                                              ])
        # print(f'\t[{i + 1} / {ncombos}] Subject {n + 1} / {nsubjects}: {default_timer() - t1:.1f} secs')
    print(f'Finished combo {i + 1} / {len(param_combi)}: {default_timer() - t0:.1f} secs')
    return model_fit

if __name__ == '__main__':

    # combo_id = int(sys.argv[1])
    combo_id = 5
    print(f'Combo ID: {combo_id}')

    combos = list(prod(range(len(modellist)), range(len(modellist))))
    ncombos = len(combos)

    run, simu = combos[combo_id]
    run_model = modellist[run]
    simu_model = modellist[simu]

    # print(f'[run {run + 1} / {len(modellist)}] simu {simu + 1} / {len(modellist)}')

    curr_parameter = [eval(x) for x in paramlist[simu]]

    if nparams[simu] == 2:
        param_combi = list(prod(list(curr_parameter[0]), list(curr_parameter[1])))
    elif nparams[simu] == 3:
        param_combi = list(prod(list(curr_parameter[0]), list(curr_parameter[1]), list(curr_parameter[2])))
    else:
        param_combi = list(prod(list(curr_parameter[0]), list(curr_parameter[1]), list(curr_parameter[2]), list(curr_parameter[3])))

    # with Pool(25) as pool:
    #     result = list(pool.map(loop, range(len(param_combi))))
    result = [None] * len(param_combi)
    for i in range(len(param_combi)):
        result[i] = loop(i)
    print('Simulation finished')

    df = pd.concat(result, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]

    path_ = f'/fast/users/ptasczle_c/work/Dropbox/confidence/ConfLearning/results/model_recov_r{run}_s{simu}.pkl'

    df.to_pickle(f'/fast/users/ptasczle_c/work/Dropbox/confidence/ConfLearning/results/model_recov2/model_recov_r{run}_s{simu}.pkl', protocol=4)
    print(f'Saved to path {path_}')