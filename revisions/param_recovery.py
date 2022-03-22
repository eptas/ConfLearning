import os
import numpy as np
import pandas as pd

from pathlib import Path
from itertools import product as prod

from ConfLearning.models.rl_simple_simchoice import RescorlaConfBaseGen
from ConfLearning.models.maximum_likelihood import ParameterFit
from ConfLearning.recovery.bandit import BanditMoney
from ConfLearning.recovery.gen_design import GenDesign

generate_data = True
use_10 = True

fitting = ParameterFit()

cwd = Path.cwd()
path_data = os.path.join(cwd, '../results/fittingData/')
winning_model = RescorlaConfBaseGen
real_win_model = 'MonoUnspec'
sim_win_id = 4

ndatasets = 100
nblocks = 110 if use_10 else 11
nphases = 3
ntrials_phase_max = 57
nbandits = 5

fitData = pd.read_pickle(os.path.join(path_data, 'fittingData_' + real_win_model + '_simchoice.pkl'))

alpha = np.arange(0, 1.00001, 0.25)
beta = np.arange(round(fitData.BETA.min(), 3), 2.00001, step=(2 - round(fitData.BETA.min(), 3)) / 4)
alpha_c = np.arange(0, 1.00001, 0.25)
gamma = np.arange(0, 10.00001, 2.5)


real_paras = ['ALPHA', 'BETA', 'ALPHA_C', 'GAMMA']  # keep this order due to rl_simple para order

choice = np.full((ndatasets, len(alpha), len(beta), len(alpha_c), len(gamma), nblocks, nphases, ntrials_phase_max), np.nan)
out_val = np.full((ndatasets, len(alpha), len(beta), len(alpha_c), len(gamma), nblocks, nphases, ntrials_phase_max), np.nan)
conf_val = np.full((ndatasets, len(alpha), len(beta), len(alpha_c), len(gamma), nblocks, nphases, ntrials_phase_max), np.nan)


design_data = ['stim_left', 'stim_right', 'history_constraint']

for d, des in enumerate(design_data):
    if use_10:
        locals()[design_data[d]] = np.load(os.path.join(cwd, (design_data[d] + "_10.npy")))
    else:
        locals()[design_data[d]] = np.load(os.path.join(cwd, (design_data[d] + ".npy")))


# initialize for model fitting

la, ua = 0, 1
lb, ub = 0, 2
lan, uan = 0, 1
lg, ug = 0, 10

grid_alpha = np.arange(0, 1.00001, 0.25)
grid_beta = np.arange(round(fitData.BETA.min(), 3), 2.00001, step=((2 - round(fitData.BETA.min(), 3)) / 4))   # (2 - round(fitData.BETA.min(), 3)) / 4)
grid_alpha_c = np.arange(0, 1.00001, 0.25)
grid_gamma = np.hstack((0, np.arange(0, 10.00001, 2.5)))

bounds = np.c_[np.array([la, lb, lan, lg]), np.array([ua, ub, uan, ug])]
expect = (np.array([ua, ub, uan, ug]) - np.array([la, lb, lan, lg])) / 2
grid_range = [grid_alpha, grid_beta, grid_alpha_c, grid_gamma]


# initialize outcome variables

probab_choice = np.full((len(real_paras), ndatasets, nblocks, nphases, ntrials_phase_max), np.nan)
saveParameters, saveFitting, saveChoiceProbab = None, None, None

parafit = np.full((len(alpha), len(beta), len(alpha_c), len(gamma), ndatasets, len(real_paras)), np.nan)
negll = np.full((len(alpha), len(beta), len(alpha_c), len(gamma), ndatasets), np.nan)
AIC = np.full((len(alpha), len(beta), len(alpha_c), len(gamma), ndatasets), np.nan, float)
BIC = np.full((len(alpha), len(beta), len(alpha_c), len(gamma), ndatasets), np.nan)

if generate_data == True:

    bandit = BanditMoney()
    GenDesign.factor = 10 if use_10 else 1
    design = GenDesign()

    for pa, para in enumerate(list(prod(alpha, beta, alpha_c, gamma))):

        parameter = para
        sim_model = winning_model(*parameter)

        al_id = list(alpha).index(para[0])
        be_id = list(beta).index(para[1])
        ac_id = list(alpha_c).index(para[2])
        ga_id = list(gamma).index(para[3])

        for i in np.arange(0, ndatasets):

            np.random.seed(i)
            design.generate()
            sim_model.noise = np.random.rand(1)

            print("Simulating dataset " + str(pa + 1) + " out of " + str(len(list(prod(alpha, beta, alpha_c, gamma)))) + ": subject " + str(i + 1) + " out of 100")

            for b in range(nblocks):

                sim_model.values = np.full(nbandits, 0, float)

                bandit.reset_outcome_history()
                bandit.set_outcome_schedule(design.outcome_schedule[b], design.outcome_base[b], design.outcome_diff[b])

                for p in range(nphases):
                    for t, tri in enumerate(np.where(~np.isnan(eval("stim_left")[i, b, p]))[0]):

                        sim_model.get_current_trial(tri)
                        sim_model.stims = np.array([int(eval("stim_left")[i, b, p, tri]), int(eval("stim_right")[i, b, p, tri])])

                        cp = sim_model.get_choice_probab()
                        choice[i, al_id, be_id, ac_id, ga_id, b, p, t], choice_index = sim_model.simulated_choice()

                        sim_model.stim_chosen = int(choice[i, al_id, be_id, ac_id, ga_id, b, p, t])

                        out = bandit.sample(int(choice[i, al_id, be_id, ac_id, ga_id, b, p, t]), ignore_history_constraints=eval("history_constraint")[i, b, p, tri])
                        out_val[i, al_id, be_id, ac_id, ga_id, b, p, t] = out if (p != 1) else np.nan

                        conf_val[i, al_id, be_id, ac_id, ga_id, b, p, t] = sim_model.simulated_confidence(choice_index)

                        sim_model.update(out_val[i, al_id, be_id, ac_id, ga_id, b, p, t], conf_val[i, al_id, be_id, ac_id, ga_id, b, p, t])

sim_variables = ['choice', 'out_val', 'conf_val']

simu_data = ['para_choice', 'para_out', 'para_conf']
para_choice, para_out, para_conf = None, None, None

print('Load data')

for v, var in enumerate(sim_variables):

    print(f'v = {v + 1} / {len(sim_variables)}')

    if use_10:
        if (generate_data == True):
            np.save(os.path.join(cwd, f'{simu_data[v]}_10'), eval(var))
        else:
            locals()[simu_data[v]] = np.load(os.path.join(cwd, simu_data[v] + '_10.npy'))
    else:
        if (generate_data == True):
            np.save(os.path.join(cwd, f'{simu_data[v]}'), eval(var))
        else:
            locals()[simu_data[v]] = np.load(os.path.join(cwd, simu_data[v] + '.npy'))

print('End load data')


def recov_params(paras, running_model, s, simulation_model, return_cp=False):

    winModel = running_model(*paras)

    alp_id = eval(simulation_model[0])
    bet_id = eval(simulation_model[1])
    alc_id = eval(simulation_model[2])
    gam_id = eval(simulation_model[3])

    negLogL = 0

    if return_cp:
        choiceprob = np.full((nblocks, nphases, ntrials_phase_max), np.nan)

    for b in range(nblocks):

        winModel.values = np.full(nbandits, 0, float)

        for p in range(nphases):
            for t, tri in enumerate(np.where(~np.isnan(eval("stim_left")[s, b, p]))[0]):

                winModel.get_current_trial(tri)

                winModel.stims = np.array([int(eval("stim_left")[s, b, p, tri]), int(eval("stim_right")[s, b, p, tri])])
                winModel.stim_chosen = int(para_choice[s, alp_id, bet_id, alc_id, gam_id, b, p, t])
                cp = winModel.get_choice_probab()

                if return_cp:
                    choiceprob[b, p, t] = cp

                negLogL -= np.log(np.maximum(cp, 1e-8))

                winModel.update(para_out[s, alp_id, bet_id, alc_id, gam_id, b, p, t], para_conf[s, alp_id, bet_id, alc_id, gam_id, b, p, t])

    return (negLogL, choiceprob) if (return_cp == True) else negLogL


if __name__ == '__main__':

    for par, paramet in enumerate(list(prod(alpha, beta, alpha_c, gamma))):

        print(f'par = {par + 1} / {len(list(prod(alpha, beta, alpha_c, gamma)))}')

        print('Simulated parameters:', paramet)

        a_id = list(alpha).index(paramet[0])
        b_id = list(beta).index(paramet[1])
        aco_id = list(alpha_c).index(paramet[2])
        g_id = list(gamma).index(paramet[3])


        for i in np.arange(0, ndatasets):

            np.random.seed(i)

            print("Fitting dataset " + str(par + 1) + " out of " + str(len(list(prod(alpha, beta, alpha_c, gamma)))) + ": subject " + str(i + 1) + " out of 100")

            simu_id = [str(a_id), str(b_id), str(aco_id), str(g_id)]

            fitting.set_model(i, ndatasets, winning_model, recov_params, 4, simu_id)
            fitting.local_minima(expect, bounds, grid_range, grid_multiproc=False)

            for p in range(len(real_paras)):
                parafit[a_id, b_id, aco_id, g_id, i, p] = min(bounds[p][1], max(bounds[p][0], fitting.data[i, p]))

            negll[a_id, b_id, aco_id, g_id, i], probab_choice[a_id, b_id, aco_id, g_id, i] = recov_params(fitting.data[i], winning_model, i, simu_id, return_cp=True)
            nsamples = np.sum(~np.isnan(probab_choice[a_id, b_id, aco_id, g_id, i]))

            AIC[a_id, b_id, aco_id, g_id, i], BIC[a_id, b_id, aco_id, g_id, i] = fitting.model_fit(negll[a_id, b_id, aco_id, g_id, i], nsamples)

            if i == (ndatasets - 1):

                parameter_fit = pd.DataFrame(data={"ALPHA_" + str(a_id) + 'a_' + str(b_id) + 'b_' + str(aco_id) + 'c_' + str(g_id) + 'g': parafit[a_id, b_id, aco_id, g_id, :, 0],
                                                   "BETA_" + str(a_id) + 'a_' + str(b_id) + 'b_' + str(aco_id) + 'c_' + str(g_id) + 'g': parafit[a_id, b_id, aco_id, g_id, :, 1],
                                                   "ALPH_N_" + str(a_id) + 'a_' + str(b_id) + 'b_' + str(aco_id) + 'c_' + str(g_id) + 'g': parafit[a_id, b_id, aco_id, g_id, :, 2],
                                                   "GAMMA_" + str(a_id) + 'a_' + str(b_id) + 'b_' + str(aco_id) + 'c_' + str(g_id) + 'g': parafit[a_id, b_id, aco_id, g_id, :, 3]
                                                   },
                                             columns=["ALPHA_" + str(a_id) + 'a_' + str(b_id) + 'b_' + str(aco_id) + 'c_' + str(g_id) + 'g',
                                                      "BETA_" + str(a_id) + 'a_' + str(b_id) + 'b_' + str(aco_id) + 'c_' + str(g_id) + 'g',
                                                      "ALPH_N_" + str(a_id) + 'a_' + str(b_id) + 'b_' + str(aco_id) + 'c_' + str(g_id) + 'g',
                                                      "GAMMA_" + str(a_id) + 'a_' + str(b_id) + 'b_' + str(aco_id) + 'c_' + str(g_id) + 'g'
                                                      ])
                saveParameters = pd.concat([saveParameters, parameter_fit], axis=1)

                model_fit = pd.DataFrame(data={"AIC_" + str(a_id) + 'a_' + str(b_id) + 'b_' + str(aco_id) + 'c_' + str(g_id) + 'g': AIC[a_id, b_id, aco_id, g_id, :],
                                               "BIC_" + str(a_id) + 'a_' + str(b_id) + 'b_' + str(aco_id) + 'c_' + str(g_id) + 'g': BIC[a_id, b_id, aco_id, g_id, :],
                                               "NEGLL_" + str(a_id) + 'a_' + str(b_id) + 'b_' + str(aco_id) + 'c_' + str(g_id) + 'g': negll[a_id, b_id, aco_id, g_id, :]
                                               },
                                         columns=["AIC_" + str(a_id) + 'a_' + str(b_id) + 'b_' + str(aco_id) + 'c_' + str(g_id) + 'g',
                                                  "BIC_" + str(a_id) + 'a_' + str(b_id) + 'b_' + str(aco_id) + 'c_' + str(g_id) + 'g',
                                                  "NEGLL_" + str(a_id) + 'a_' + str(b_id) + 'b_' + str(aco_id) + 'c_' + str(g_id) + 'g'
                                                  ])

                saveFitting = pd.concat([saveFitting, model_fit], axis=1)

                choice_probability = pd.DataFrame(data={"cp_" + str(a_id) + 'a_' + str(b_id) + 'b_' + str(aco_id) + 'c_' + str(g_id) + 'g': probab_choice[a_id, b_id, aco_id, g_id, :][~np.isnan(probab_choice[a_id, b_id, aco_id, g_id, :])]},
                                                  columns=["cp_" + str(a_id) + 'a_' + str(b_id) + 'b_' + str(aco_id) + 'c_' + str(g_id) + 'g'])

                saveChoiceProbab = pd.concat([saveChoiceProbab, choice_probability], axis=1)

        if par == (len(real_paras) - 1):

            if use_10:
                pd.concat([saveParameters, saveFitting], axis=1).to_pickle("fittingData_para_simu_10.pkl", protocol=4)
                saveChoiceProbab.to_pickle("choiceProbab_para_simu_10.pkl", protocol=4)
            else:
                pd.concat([saveParameters, saveFitting], axis=1).to_pickle("fittingData_para_simu.pkl", protocol=4)
                saveChoiceProbab.to_pickle("choiceProbab_para_simu.pkl", protocol=4)
