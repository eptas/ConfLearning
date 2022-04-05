import os
import numpy as np
import pandas as pd

from pathlib import Path
from itertools import product as prod
from scipy.stats import spearmanr as spear

from ConfLearning.models.rl_simple_simchoice import RescorlaConfBaseGen
from ConfLearning.models.maximum_likelihood import ParameterFit
from ConfLearning.recovery.bandit import BanditMoney
from ConfLearning.recovery.gen_design import GenDesign

fitting = ParameterFit()
bandit = BanditMoney()
GenDesign.factor = 10
design = GenDesign()

cwd = Path.cwd()
path_data = os.path.join(cwd, '../results/fittingData/')
winning_model = RescorlaConfBaseGen
real_win_model = 'MonoUnspec'
sim_win_id = 4

design_id = 0
nblocks = 11
nphases = 3
ntrials_phase_max = 57
nbandits = 5

fitData = pd.read_pickle(os.path.join(path_data, 'fittingData_' + real_win_model + '_simchoice.pkl'))

alpha = np.arange(0, 1.00001, 0.25)
beta = np.arange(round(fitData.BETA.min(), 3), 2.00001, step=(2 - round(fitData.BETA.min(), 3)) / 4)
alpha_c = np.arange(0, 1.00001, 0.25)
gamma = np.arange(0, 10.00001, 2.5)

alpha_loop = np.arange(0, 1, 0.01)
beta_loop = np.arange(round(fitData.BETA.min(), 3), 2, step=(2 - round(fitData.BETA.min(), 3)) / 99)
alpha_c_loop = np.arange(0, 1, 0.01)
gamma_loop = np.arange(0, 10, 0.1)

real_paras = ['ALPHA', 'BETA', 'ALPHA_C', 'GAMMA']  # keep this order due to rl_simple para order


choice = np.full((nblocks, nphases, ntrials_phase_max), np.nan)
out_val = np.full((nblocks, nphases, ntrials_phase_max), np.nan)
conf_val = np.full((nblocks, nphases, ntrials_phase_max), np.nan)

design_data = ['stim_left', 'stim_right', 'history_constraint']

for d, des in enumerate(design_data):
    locals()[design_data[d]] = np.load(os.path.join(cwd, (design_data[d] + "_10.npy")))


# initialize for model fitting

la, ua = 0, 1
lb, ub = 0, 2
lan, uan = 0, 1
lg, ug = 0, 10

grid_alpha = np.arange(0.1, 0.51, 0.2)
grid_beta = np.arange(0.1, 0.31, 0.1)
grid_alpha_c = np.arange(0.01, 0.061, 0.05)
grid_gamma = np.hstack((0, np.arange(0.05, 0.5, 0.05)))    # np.arange(0.05, 0.5, 0.1)


bounds = np.c_[np.array([la, lb, lan, lg]), np.array([ua, ub, uan, ug])]
expect = (np.array([ua, ub, uan, ug]) - np.array([la, lb, lan, lg])) / 2
grid_range = [grid_alpha, grid_beta, grid_alpha_c, grid_gamma]

parafit = np.full((len(real_paras), len(alpha_loop), len(real_paras)), np.nan)
corr_array = np.full((len(real_paras), len(alpha_loop), len(real_paras)), np.nan)

corr_matrix = np.full(((len(alpha)**len(real_paras)), len(real_paras), len(real_paras)), np.nan)


def sim_behaviour(s, simulation_parameter):

    simulation_params = ['alp_id', 'bet_id', 'alc_id', 'gam_id', 'sweep_para', 'loop_id']
    alp_id, bet_id, alc_id, gam_id, sweep_para, loop_id = None, None, None, None, None, None

    for sim_par, sim_paras in enumerate(simulation_params):
        locals()[simulation_params[sim_par]] = eval(simulation_parameter[sim_par])

    base_list = [alpha[alp_id], beta[bet_id], alpha_c[alc_id], gamma[gam_id]]
    base_list[sweep_para] = eval(real_paras[sweep_para].lower() + '_loop')[loop_id]

    sim_parameter = base_list

    simModel = winning_model(*sim_parameter)

    np.random.seed(s)
    design.generate()

    for block in range(nblocks):

        simModel.values = np.full(nbandits, 0, float)

        bandit.reset_outcome_history()
        bandit.set_outcome_schedule(design.outcome_schedule[block], design.outcome_base[block], design.outcome_diff[block])

        for phase in range(nphases):
            for tria, trials in enumerate(np.where(~np.isnan(eval("stim_left")[design_id, block, phase]))[0]):

                simModel.noise = np.random.rand(1)

                simModel.get_current_trial(trials)
                simModel.stims = np.array([int(eval("stim_left")[design_id, block, phase, trials]), int(eval("stim_right")[design_id, block, phase, trials])])

                cp = simModel.get_choice_probab()

                choice[block, phase, tria], choice_index = simModel.simulated_choice()
                simModel.stim_chosen = int(choice[block, phase, tria])

                out = bandit.sample(int(choice[block, phase, tria]), ignore_history_constraints=eval("history_constraint")[design_id, block, phase, trials])
                out_val[block, phase, tria] = out if (phase != 1) else np.nan

                conf_val[block, phase, tria] = simModel.simulated_confidence(choice_index)

                simModel.update(out_val[block, phase, tria], conf_val[block, phase, tria])

    return choice, out_val, conf_val


def recov_params(paras, running_model, s, simulation_model, return_cp=False):

    choices, outcome_value, confidence_value = sim_behaviour(s, simulation_model)

    winModel = running_model(*paras)

    negLogL = 0

    if return_cp:
        choiceprob = np.full((nblocks, nphases, ntrials_phase_max), np.nan)

    for b in range(nblocks):

        winModel.values = np.full(nbandits, 0, float)

        for p in range(nphases):
            for t, tri in enumerate(np.where(~np.isnan(eval("stim_left")[design_id, b, p]))[0]):

                winModel.get_current_trial(tri)
                winModel.stims = np.array([int(eval("stim_left")[design_id, b, p, tri]), int(eval("stim_right")[design_id, b, p, tri])])

                cp = winModel.get_choice_probab()

                winModel.stim_chosen = int(choices[b, p, t])

                winModel.update(outcome_value[b, p, t], confidence_value[b, p, t])

                if return_cp:
                    choiceprob[b, p, t] = cp

                negLogL -= np.log(np.maximum(cp, 1e-8))

    return (negLogL, choiceprob) if (return_cp == True) else negLogL


if __name__ == '__main__':

    for pa, para in enumerate(list(prod(alpha, beta, alpha_c, gamma))):

        al_id = list(alpha).index(para[0])
        be_id = list(beta).index(para[1])
        ac_id = list(alpha_c).index(para[2])
        ga_id = list(gamma).index(para[3])

        for sw, sweep in enumerate(real_paras):
            for lo, loop in enumerate(eval(sweep.lower() + '_loop')):

                print("Fitting dataset " + str(pa + 1) + " out of " + str(len(list(prod(alpha, beta, alpha_c, gamma)))) + " : " + sweep + str(lo))

                simu_id = [str(al_id), str(be_id), str(ac_id), str(ga_id), str(sw), str(lo)]

                base_parameters = [alpha[al_id], beta[be_id], alpha_c[ac_id], gamma[ga_id]]
                base_parameters[sw] = lo

                fitting.set_model(design_id, 1, winning_model, recov_params, 4, simu_id)
                fitting.local_minima(expect, bounds, grid_range, grid_multiproc=False)

                for repa in range(len(real_paras)):

                    parafit[sw, lo, repa] = min(bounds[repa][1], max(bounds[repa][0], fitting.data[design_id, repa]))
                    corr_array[sw, lo, repa] = base_parameters[repa]

        for sico, simcor in enumerate(real_paras):
            for fico, fitcor in enumerate(real_paras):

                rho, pval = spear(parafit[sico, :, fico], corr_array[sico, :, fico])
                corr_matrix[pa, sico, fico] = rho

    np.save('correlation_matrix.npy', corr_matrix)
