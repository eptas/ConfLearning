#!/fast/home/users/bawolf_c/python/bin/python3 -u
from concurrent.futures import ProcessPoolExecutor as Pool
from timeit import default_timer
import os
import numpy as np
import pandas as pd

from pathlib import Path
from itertools import product as prod
import sys
HOME = os.path.expanduser("~")
sys.path.extend([os.path.join(HOME, 'work/Dropbox/confidence/')])
from ConfLearning.models.rl_simple import Rescorla, RescorlaZero as RescorlaDeval, RescorlaPerseveration
from ConfLearning.models.rl_simple_choice_simchoice import RescorlaChoiceMono
from ConfLearning.models.rl_simple_simchoice import RescorlaConfBase, RescorlaConfBaseGen

from ConfLearning.models.maximum_likelihood import ParameterFit
from ConfLearning.revision2.bandit import BanditMoney
from ConfLearning.revision2.gen_design_sim import GenDesign

# initialize data simulation

fitting = ParameterFit()
GenDesign.factor = 1
design = GenDesign()

nsubjects = 250
# nsubjects = 4
n_datasets = nsubjects

nblocks = 11
nphases = 3
ntrials_phase_max = 57
nbandits = 5

modellist = [Rescorla, RescorlaDeval, RescorlaChoiceMono, RescorlaConfBase, RescorlaConfBaseGen, RescorlaPerseveration]
# modellist = [Rescorla, RescorlaDeval]
model_names = ['Static', 'Deval', 'Choice', 'ConfSpec', 'ConfUnspec', 'Perseveration']
paramlist = [['alpha', 'beta'], ['alpha', 'beta', 'alpha_n'], ['alpha', 'beta', 'lambd'], *[['alpha', 'beta', 'alpha_c', 'gamma'] for _ in range(2)], ['alpha', 'beta', 'eta']]   # for behav fitting dataframe

var_list = ['stim_left', 'stim_right', 'history_constraint']
stim_left, stim_right, history_constraint = None, None, None


try:
    # stim_left, stim_right, history_constraint = [np.load(os.path.join(HOME, 'work/Dropbox/confidence/ConfLearning/revisions/', f'{v}_10.npy')) for v in var_list]
    stim_left, stim_right, history_constraint = [np.load(os.path.join(HOME, 'work/Dropbox/confidence/ConfLearning/revisions/', f'{v}.npy')) for v in var_list]
except:
    cwd = Path.cwd()
    for v, variable in enumerate(var_list):
        # locals()[variable] = np.load(os.path.join(cwd, variable + '_10.npy'))
        locals()[variable] = np.load(os.path.join(cwd, variable + '.npy'))

alpha = np.linspace(0.1, 1, 5)
beta = np.array([0.1, 0.2, 0.4, 0.8, 1.6])
alpha_n = np.linspace(0.1, 1, 5)
alpha_c = np.linspace(0.1, 1, 5)
lambd = np.exp(np.linspace(np.log(0.5), np.log(5), 5))
gamma = np.exp(np.linspace(np.log(1), np.log(100), 5))
eta = np.linspace(-1.5, 1.5, 6)

la, ua = 0, 1
lb, ub = 0, 4
lan, uan = 0, 1
lac, uac = 0, 1
ll, ul = 0, 10
lg, ug = 0, 100
le, ue = -5, 5

# grid_alpha = np.arange(0.1, 0.51, 0.2)
# grid_beta = np.arange(0.1, 0.61, 0.2)
# grid_alpha_c = np.arange(0.01, 1.01, 0.1)
# grid_gamma = np.arange(5.1)    # np.arange(0.05, 0.5, 0.1)

grid_alpha = np.arange(0.1, 0.51, 0.2)
grid_beta = np.arange(0.1, 0.51, 0.2)
grid_alpha_n = np.arange(0.1, 1.01, 0.2)
grid_alpha_c = np.arange(0.1, 1.01, 0.2)
grid_gamma = np.linspace(0, 50, 6)
grid_lambd = np.linspace(0, 5, 6)
grid_eta = np.arange(-0.5, 0.51, 0.2)
ga, gb, gan, gac, gg, gl, ge = grid_alpha, grid_beta, grid_alpha_n, grid_alpha_c, grid_gamma, grid_lambd, grid_eta

grid_range = [
    [grid_alpha, grid_beta],
    [grid_alpha, grid_beta, grid_alpha_n],
    [grid_alpha, grid_beta, grid_gamma],
    *[[grid_alpha, grid_beta, grid_alpha_c, grid_gamma] for _ in range(2)],
    [grid_alpha, grid_beta, grid_eta]
]

# bounds = [np.c_[np.array([la, lb]), np.array([ua, ub])],
#           np.c_[np.array([la, lb, lac]), np.array([ua, ub, uac])],
#           np.c_[np.array([la, lb, lg]), np.array([ua, ub, ug])],
#           *[np.c_[np.array([la, lb, lac, lg]), np.array([ua, ub, uac, ug])] for _ in range(2)],
#           np.c_[np.array([la, lb, le]), np.array([ua, ub, ue])]
# ]
bounds = [np.c_[np.array([la, lb]), np.array([ua, ub])],
          np.c_[np.array([la, lb, lan]), np.array([ua, ub, uan])],
          np.c_[np.array([la, lb, ll]), np.array([ua, ub, ul])],
          *[np.c_[np.array([la, lb, lac, lg]), np.array([ua, ub, uac, ug])] for _ in range(2)],
          np.c_[np.array([la, lb, le]), np.array([ua, ub, ue])]
]

expect = [(np.array([ga[0], gb[0]]) + np.array([ga[-1], gb[-1]])) / 2,
          (np.array([ga[0], gb[0], gan[0]]) + np.array([ga[-1], gb[-1], gan[-1]])) / 2,
          (np.array([ga[0], gb[0], gl[0]]) + np.array([ga[-1], gb[-1], gl[-1]])) / 2,
          *[(np.array([ga[0], gb[0], gac[0], gg[0]]) + np.array([ga[-1], gb[-1], gac[-1], gg[-1]])) / 2 for _ in range(2)],
          (np.array([ga[0], gb[0], ge[0]]) + np.array([ga[-1], gb[-1], ge[-1]])) / 2
]


nparams = [2, 3, 3, 4, 4, 3]

# probab_choice = np.full((len(modellist), len(modellist), len(alpha)**max(nparams), nsubjects, n_datasets, nblocks, nphases, ntrials_phase_max), np.nan)
# negll = np.full((len(modellist), len(modellist), len(alpha)**max(nparams), nsubjects), np.nan)
# AIC, BIC = np.full((len(modellist), len(modellist), len(alpha)**max(nparams), nsubjects), np.nan, float), np.full((len(modellist), len(modellist), nsubjects), np.nan)


choice = np.full((n_datasets, nblocks, nphases, ntrials_phase_max), np.nan)
out_val = np.full((n_datasets, nblocks, nphases, ntrials_phase_max), np.nan)
conf_val = np.full((n_datasets, nblocks, nphases, ntrials_phase_max), np.nan)

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


def fit(parame, running_model, s, sim_params, choices, outcome_value, confidence_value, return_cp=False):

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

                modelling.stim_chosen = int(choices[s, b, p, t])
                cp = modelling.get_choice_probab()

                modelling.update(outcome_value[s, b, p, t], confidence_value[s, b, p, t])

                if return_cp:
                    choiceprob[b, p, t] = cp

                negLogL -= np.log(np.maximum(cp, 1e-8))

    return (negLogL, choiceprob) if (return_cp == True) else negLogL

def loop(i):

    t0 = default_timer()

    param_ids, params = param_combi[i], [curr_parameter[j][p] for j, p in enumerate(param_combi[i])]

    choices, outcome_value, confidence_value = sim_data(gen_id, params)
    # print(f'\t\tSimulation took {default_timer() - ts:.1f} secs')

    d = pd.DataFrame(index=range(nsubjects))
    for j, p in enumerate(paramlist[gen_id]):
        d[f'{p}_id'] = param_ids[j]
        d[p] = params[j]
    d['subject'] = range(nsubjects)

    for n in range(nsubjects):

        # t1 = default_timer()

        # print(f'\tSubject {n + 1} / {nsubjects}')

        fitting.set_model(n, nsubjects, fit_model, fit, nparams[fit_id])
        fitting.local_minima(expect[fit_id], bounds[fit_id], grid_range[fit_id], grid_multiproc=False, verbose=False, args=[choices, outcome_value, confidence_value])

        negll, probab_choice = fit(fitting.data[n], fit_model, n, gen_model, choices, outcome_value, confidence_value, return_cp=True)
        # fitting.fit_model(para, fitting.model, fitting.subj, fitting.sim_model, choices, outcome_value, confidence_value)
        # fitting.fit_model(fitting.data[n], fitting.model, fitting.subj, fitting.sim_model, choices, outcome_value, confidence_value)
        nsamples = np.sum(~np.isnan(probab_choice))

        AIC, BIC = fitting.model_fit(negll, nsamples)
        d.loc[d.subject == n, 'AIC'] = AIC
        d.loc[d.subject == n, 'BIC'] = BIC

        # print(f'\t[{i + 1} / {ncombos}] Subject {n + 1} / {nsubjects}: {default_timer() - t1:.1f} secs')
    print(f'Finished combo {i + 1} / {len(param_combi)}: {default_timer() - t0:.1f} secs')
    return d

if __name__ == '__main__':

    if len(sys.argv) > 1:
        combo_id = int(sys.argv[1])
    else:
        combo_id = 0
    print(f'Combo ID: {combo_id}')

    combos = list(prod(range(len(modellist)), range(len(modellist))))
    ncombos = len(combos)

    gen_id, fit_id = combos[combo_id]
    gen_model, fit_model = modellist[gen_id], modellist[fit_id]
    gen_model_name, fit_model_name = model_names[gen_id], model_names[fit_id]

    # print(f'[fit_id {fit_id + 1} / {len(modellist)}] gen_id {gen_id + 1} / {len(modellist)}')

    curr_parameter = [eval(x) for x in paramlist[gen_id]]

    if nparams[gen_id] == 2:
        param_combi = list(prod(range(len(curr_parameter[0])), range(len(curr_parameter[1]))))
    elif nparams[gen_id] == 3:
        param_combi = list(prod(range(len(curr_parameter[0])), range(len(curr_parameter[1])), range(len(curr_parameter[2]))))
    else:
        param_combi = list(prod(range(len(curr_parameter[0])), range(len(curr_parameter[1])), range(len(curr_parameter[2])), range(len(curr_parameter[3]))))

    with Pool(25) as pool:
        result = list(pool.map(loop, range(len(param_combi))))
    # result = [None] * len(param_combi)
    # for i in range(len(param_combi)):
    #     result[i] = loop(i)
    print('Simulation finished')
    df = pd.concat(result).reset_index(drop=True)
    df['gen_model_id'] = gen_id
    df['gen_model'] = gen_model_name
    df['fit_model_id'] = fit_id
    df['fit_model'] = fit_model_name
    df = df[list(df.columns[-4:]) + list(df.columns[:-4])]

    path_ = os.path.join(HOME, f'work/Dropbox/confidence/ConfLearning/results/model_recov_g{gen_id}_f{fit_id}.pkl.gz')
    df.to_pickle(path_, protocol=4)
    print(f'Saved to path {path_}')