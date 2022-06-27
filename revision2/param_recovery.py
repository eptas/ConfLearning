#!/fast/home/users/ptasczle_c/python/bin/python3 -u

from concurrent.futures import ProcessPoolExecutor as Pool
from timeit import default_timer
import os
import numpy as np
import pandas as pd

from pathlib import Path
from itertools import product as prod
# from scipy.stats import spearmanr as spear
from scipy.stats import pearsonr, spearmanr, theilslopes
import socket
from pingouin import corr
import sys
import os
HOME = os.path.expanduser("~")
sys.path.extend([os.path.join(HOME, 'work/Dropbox/confidence/')])
from ConfLearning.models.rl_simple_simchoice import RescorlaConfBaseGen
from ConfLearning.models.maximum_likelihood import ParameterFit
from ConfLearning.recovery.bandit import BanditMoney
from ConfLearning.recovery.gen_design_sim import GenDesign

use_10 = False

fitting = ParameterFit()
bandit = BanditMoney()
GenDesign.factor = 10 if use_10 else 1
design = GenDesign()
design_id = 0
np.random.seed(design_id)
design.generate()

winning_model = RescorlaConfBaseGen
real_win_model = 'MonoUnspec'
sim_win_id = 4

nblocks = 11
nphases = 3
ntrials_phase_max = 57
nbandits = 5

design_data = ['stim_left', 'stim_right', 'history_constraint']
try:
    path_data = os.path.join(HOME, 'work/Dropbox/confidence/ConfLearning/results/fittingData/')
    fitData = pd.read_pickle(os.path.join(path_data, 'fittingData_' + real_win_model + '_simchoice.pkl'))
    if use_10:
        stim_left, stim_right, history_constraint = [np.load(os.path.join(HOME, 'work/Dropbox/confidence/ConfLearning/revisions/', f'{d}_10.npy')) for d in design_data]
    else:
        stim_left, stim_right, history_constraint = [np.load(os.path.join(HOME, 'work/Dropbox/confidence/ConfLearning/revisions/', f'{d}.npy')) for d in design_data]
except:
    cwd = Path.cwd()
    path_data = os.path.join(cwd, '../results/fittingData/')
    fitData = pd.read_pickle(os.path.join(path_data, 'fittingData_' + real_win_model + '_simchoice.pkl'))
    for d, des in enumerate(design_data):
        if use_10:
            locals()[design_data[d]] = np.load(os.path.join(cwd, (design_data[d] + "_10.npy")))
        else:
            locals()[design_data[d]] = np.load(os.path.join(cwd, (design_data[d] + ".npy")))

nsweep = 250
real_paras = ['ALPHA', 'BETA', 'ALPHA_C', 'GAMMA']  # keep this order due to rl_simple para order


alpha = np.linspace(0.1, 1, 5)
beta = np.array([0.1, 0.2, 0.4, 0.8, 1.6])
alpha_c = np.linspace(0.1, 1, 5)
gamma = np.exp(np.linspace(0, 4, 5))
# alpha = np.linspace(0.1, 1, 4)
# beta = np.exp(np.linspace(np.log(0.1), np.log(2), 4))
# alpha_c = np.linspace(0.1, 1, 4)
# gamma = np.exp(np.linspace(0, 4, 4))

alpha_loop = np.linspace(0.01, 1, nsweep)
beta_loop = np.linspace(0.02, 2, nsweep)
alpha_c_loop = np.linspace(0, 1, nsweep)
gamma_loop = np.linspace(0, 10, nsweep)

grid_alpha = np.arange(0.1, 0.51, 0.2)
grid_beta = np.arange(0.1, 0.51, 0.2)
grid_alpha_c = np.arange(0.1, 1.01, 0.2)
grid_gamma = np.linspace(0, 50, 6)

la, ua = 0, 1
lb, ub = 0, 2
lac, uac = 0, 1
lg, ug = 0, 100
bounds = np.c_[np.array([la, lb, lac, lg]), np.array([ua, ub, uac, ug])]
expect = np.array([(grid_alpha[0] + grid_alpha[-1])/2, (grid_beta[0] + grid_beta[-1])/2,
                   (grid_alpha_c[0] + grid_alpha_c[-1])/2, (grid_gamma[0] + grid_gamma[-1])/2])
grid_range = [grid_alpha, grid_beta, grid_alpha_c, grid_gamma]


choice = np.full((nblocks, nphases, ntrials_phase_max), np.nan)
out_val = np.full((nblocks, nphases, ntrials_phase_max), np.nan)
conf_val = np.full((nblocks, nphases, ntrials_phase_max), np.nan)

parafit = np.full((len(real_paras), len(alpha_loop), len(real_paras)), np.nan)
corr_array = np.full((len(real_paras), len(alpha_loop), len(real_paras)), np.nan)


def sim_behaviour(simulation_parameter):

    # simulation_params = ['alp_id', 'bet_id', 'alc_id', 'gam_id', 'sweep_para', 'loop_id']
    # alp_id, bet_id, alc_id, gam_id, sweep_para, loop_id = None, None, None, None, None, None
    # for sim_par, sim_paras in enumerate(simulation_params):
    #     locals()[simulation_params[sim_par]] = eval(simulation_parameter[sim_par])
    # alp_id, bet_id, alc_id, gam_id, sweep_para, loop_id = [eval(simulation_parameter[i]) for i in range(len(simulation_parameter))]
    #
    # base_list = [alpha[alp_id], beta[bet_id], alpha_c[alc_id], gamma[gam_id]]
    # base_list[sweep_para] = eval(real_paras[sweep_para].lower() + '_loop')[loop_id]
    #
    # sim_parameter = base_list

    simModel = winning_model(*simulation_parameter)

    for block in range(nblocks):

        simModel.values = np.full(nbandits, 0, float)

        bandit.reset_outcome_history()
        bandit.set_outcome_schedule(design.outcome_schedule[block], design.outcome_base[block], design.outcome_diff[block])

        for phase in range(nphases):
            for tria, trials in enumerate(np.where(~np.isnan(eval("stim_left")[design_id, block, phase]))[0]):
            # for tria, trials in enumerate(np.where(~design.design[(design.design.block == block) & (design.design.phase == phase)].stimulus_left.isna())[0]):
            #     cond = (design.design.block == block) & (design.design.phase == phase) & (design.design.trial_phase == tria)
                simModel.noise = np.random.rand()

                # simModel.get_current_trial(trials)
                simModel.stims = np.array([int(eval("stim_left")[design_id, block, phase, trials]), int(eval("stim_right")[design_id, block, phase, trials])])
                # simModel.stims = np.array([design.design[cond].stimulus_left.item(), design.design[cond].stimulus_right.item()])

                simModel.get_choice_probab()
                choice[block, phase, tria], choice_index = simModel.simulated_choice()
                simModel.stim_chosen = int(choice[block, phase, tria])

                out = bandit.sample(int(choice[block, phase, tria]), ignore_history_constraints=eval("history_constraint")[design_id, block, phase, trials])
                # out = bandit.sample(int(choice[block, phase, tria]), ignore_history_constraints=design.design[cond].pre_equalshown_secondlasttrial.item())
                out_val[block, phase, tria] = out if (phase != 1) else np.nan

                conf_val[block, phase, tria] = simModel.simulated_confidence(choice_index)

                simModel.update(out_val[block, phase, tria], conf_val[block, phase, tria])

    return choice, out_val, conf_val


def recov_params(paras, running_model, s, sim_params, choices, outcome_value, confidence_value, return_cp=False):
    # ty = default_timer()
    winModel = running_model(*paras)

    negLogL = 0

    if return_cp:
        choiceprob = np.full((nblocks, nphases, ntrials_phase_max), np.nan)

    for b in range(nblocks):

        winModel.values = np.full(nbandits, 0, float)

        for p in range(nphases):
            for t, tri in enumerate(np.where(~np.isnan(eval("stim_left")[design_id, b, p]))[0]):
            # for t, tri in enumerate(np.where(~design.design[(design.design.block == b) & (design.design.phase == p)].stimulus_left.isna())[0]):
                # cond = (design.design.block == b) & (design.design.phase == p) & (design.design.trial_phase == tri)

                # winModel.get_current_trial(tri)
                winModel.stims = np.array([int(eval("stim_left")[design_id, b, p, tri]), int(eval("stim_right")[design_id, b, p, tri])])
                # winModel.stims = np.array([design.design[cond].stimulus_left.item(), design.design[cond].stimulus_right.item()])

                winModel.stim_chosen = int(choices[b, p, t])
                cp = winModel.get_choice_probab()

                winModel.update(outcome_value[b, p, t], confidence_value[b, p, t])

                if return_cp:
                    choiceprob[b, p, t] = cp

                negLogL -= np.log(np.maximum(cp, 1e-8))
    # print(f'\tty: {1000*(default_timer() - ty):.1f} ms')
    return (negLogL, choiceprob) if (return_cp == True) else negLogL


def loop(i):

    print(f'Starting combo {i + 1} / {ncombos}')

    np.random.seed(design_id)

    para = combos[i]

    al_id = list(alpha).index(para[0])
    be_id = list(beta).index(para[1])
    ac_id = list(alpha_c).index(para[2])
    ga_id = list(gamma).index(para[3])
    t0 = default_timer()
    cm_pearson = np.full((len(real_paras), len(real_paras)), np.nan)
    cm_pearson_robust = np.full((len(real_paras), len(real_paras)), np.nan)
    cm_spearman = np.full((len(real_paras), len(real_paras)), np.nan)
    cm_slope_robust = np.full((len(real_paras), len(real_paras)), np.nan)
    negll = np.full((len(real_paras), nsweep), np.nan)
    negll_true = np.full((len(real_paras), nsweep), np.nan)
    for sw, sweep in enumerate(real_paras):
        t1 = default_timer()
        sweep_range = eval(sweep.lower() + '_loop')
        parafit = np.full((len(sweep_range), len(real_paras)), np.nan)
        for lo, loop in enumerate(sweep_range):
            t2 = default_timer()

            base_parameters = [alpha[al_id], beta[be_id], alpha_c[ac_id], gamma[ga_id]]
            base_parameters[sw] = loop
            # print('Start simulation')
            choices, outcome_value, confidence_value = sim_behaviour(base_parameters)
            # print('Stop simulation')

            # fitting.set_model(design_id, 1, winning_model, recov_params, 4)
            fitting.set_model(0, 1, winning_model, recov_params, 4)
            # print(base_parameters)
            # tx = default_timer()
            fitting.local_minima(expect, bounds, grid_range, grid_multiproc=False, verbose=False, args=[choices, outcome_value, confidence_value])
            # print(f'\ttx: {default_timer() - tx:.1f} secs')
            negll[sw, lo] = fitting.negll
            negll_true[sw, lo] = fitting.run_model(base_parameters, fitting.model, fitting.subj, fitting.sim_model, choices, outcome_value, confidence_value)
            # fitting.run_model(base_parameters, fitting.model, fitting.subj, fitting.sim_model, choices, outcome_value, confidence_value)

            for repa in range(len(real_paras)):
                # parafit[lo, repa] = min(bounds[repa][1], max(bounds[repa][0], fitting.data[design_id, repa]))
                parafit[lo, repa] = min(bounds[repa][1], max(bounds[repa][0], fitting.data[0, repa]))
            # print(f"\t\t[{i + 1} / {ncombos}] [{sw + 1} / {len(real_paras)}] Most inner Loop {lo + 1} / {len(eval(sweep.lower() + '_loop'))}: {default_timer() - t2:.1f} secs")
        cm_pearson[sw] = [pearsonr(sweep_range, row)[0] for row in parafit.T]
        cm_pearson_robust[sw] = [corr(sweep_range, row, method='bicor').r.item() for row in parafit.T]
        cm_spearman[sw] = [spearmanr(sweep_range, row)[0] for row in parafit.T]
        cm_slope_robust[sw] = [theilslopes(row, sweep_range)[0] for row in parafit.T]

        print(f'\t[{i + 1} / {ncombos}] Inner Loop {sw + 1} / {len(real_paras)}: {default_timer() - t1:.1f} secs')

    np.save(os.path.join(HOME, f"work/Dropbox/confidence/ConfLearning/results/param_recov/cm{('', '10')[use_10]}_pearson_design{design_id:02g}_{i:03g}.npy"), cm_pearson)
    np.save(os.path.join(HOME, f"work/Dropbox/confidence/ConfLearning/results/param_recov/cm{('', '10')[use_10]}_pearson_robust_design{design_id:02g}_{i:03g}.npy"), cm_pearson_robust)
    np.save(os.path.join(HOME, f"work/Dropbox/confidence/ConfLearning/results/param_recov/cm{('', '10')[use_10]}_spearman_design{design_id:02g}_{i:03g}.npy"), cm_spearman)
    np.save(os.path.join(HOME, f"work/Dropbox/confidence/ConfLearning/results/param_recov/cm{('', '10')[use_10]}_slope_robust_design{design_id:02g}_{i:03g}.npy"), cm_slope_robust)
    np.save(os.path.join(HOME, f"work/Dropbox/confidence/ConfLearning/results/param_recov/cm{('', '10')[use_10]}_negll_design{design_id:02g}_{i:03g}.npy"), negll)
    np.save(os.path.join(HOME, f"work/Dropbox/confidence/ConfLearning/results/param_recov/cm{('', '10')[use_10]}_negll_true_design{design_id:02g}_{i:03g}.npy"), negll_true)
    print(f'Finished combo {i + 1} / {ncombos}: {default_timer() - t0:.1f} secs')
    # return cm_pearson, cm_pearson_robust, cm_spearman, cm_slope_robust, negll, negll_true

if __name__ == '__main__':

    if len(sys.argv) > 1:
        iter = int(sys.argv[1])
    else:
        iter = 3
    # iter = 3
    combo_range = np.arange(iter*25, (iter + 1)*25)
    print(f'Combo range: {combo_range[0]}-{combo_range[-1]}')

    combos = list(prod(alpha, beta, alpha_c, gamma))
    ncombos = len(combos)

    with Pool(25) as pool:
        list(pool.map(loop, combo_range))
    # for i in combo_range:
    #     loop(i)
    print('Finished!')
