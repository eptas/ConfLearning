import os
import numpy as np
import pandas as pd

from pathlib import Path
from ConfLearning.models.rl_simple_simchoice import RescorlaConfBaseGen
from ConfLearning.models.maximum_likelihood import ParameterFit
from ConfLearning.recovery.bandit import BanditMoney
from ConfLearning.recovery.design import Design
from ConfLearning.recovery.gen_design import GenDesign
from socket import gethostname

generate_data = False
use_10 = False

fitting = ParameterFit()

cwd = Path.cwd()
path_data = os.path.join(cwd, '../results/fittingData/')
path_save = '/home/matthiasg/esther/' if 'grid' in gethostname() else '/home/matteo/outsource_Dropbox/'
winning_model = RescorlaConfBaseGen
real_win_model = 'MonoUnspec'
sim_win_id = 4

# experi_matrix = pd.read_pickle(os.path.join(os.path.join(Path.cwd(), './para_experiment/'), 'experimental_sim_0.pkl'))

ndatasets = 100
nblocks = 110 if use_10 else 11
nphases = 3
ntrials_phase_max = 57      # max(experi_matrix.trial.values) + 1
nbandits = 5


alpha = np.arange(0, 1.00001, 0.01)
beta = np.arange(0, 2.01, 0.02)
alpha_c = np.arange(0, 1.00001, 0.01)
gamma = np.arange(0, 10.01, 0.1)

mALPHA, mBETA, mGAMMA, mALPHA_N = None, None, None, None

realFitData = pd.read_pickle(os.path.join(path_data, 'fittingData_' + real_win_model + '_simchoice.pkl'))

real_paras = ['ALPHA', 'BETA', 'ALPHA_C', 'GAMMA']  # keep this order due to rl_simple para order
# real_paras = ['GAMMA']  # keep this order due to rl_simple para order

choice = np.full((ndatasets, len(real_paras), len(alpha), nblocks, nphases, ntrials_phase_max), np.nan)
out_val = np.full((ndatasets, len(real_paras), len(alpha), nblocks, nphases, ntrials_phase_max), np.nan)
conf_val = np.full((ndatasets, len(real_paras), len(alpha), nblocks, nphases, ntrials_phase_max), np.nan)

if use_10:
    stim_left = np.load(os.path.join(path_save, "stim_left_10.npy"))
    stim_right = np.load(os.path.join(path_save, "stim_right_10.npy"))
    history_constraint = np.load(os.path.join(path_save, "history_constraint_10.npy"))
else:
    stim_left = np.load(os.path.join(path_save, "stim_left.npy"))
    stim_right = np.load(os.path.join(path_save, "stim_right.npy"))
    history_constraint = np.load(os.path.join(path_save, "history_constraint.npy"))


# initialize for model fitting

initial_list = [0.1, 1, 0.1, 0.1]

la, ua = 0, 1
lb, ub = 0, 2
lan, uan = 0, 1
lg, ug = 0, 100

grid_alpha = np.arange(0.1, 0.51, 0.2)
grid_beta = np.arange(0.1, 0.31, 0.1)
grid_alpha_n = np.arange(0.01, 0.061, 0.05)
grid_gamma = np.hstack((0, np.arange(0.05, 0.5, 0.05)))    # np.arange(0.05, 0.5, 0.1)

bounds = np.c_[np.array([la, lb, lan, lg]), np.array([ua, ub, uan, ug])]
expect = (np.array([ua, ub, uan, ug]) - np.array([la, lb, lan, lg])) / 2
grid_range = [grid_alpha, grid_beta, grid_alpha_n, grid_gamma]

probab_choice = np.full((len(real_paras), len(alpha), ndatasets, nblocks, nphases, ntrials_phase_max), np.nan)
saveParameters, saveFitting, saveChoiceProbab = None, None, None

parafit = np.full((len(real_paras), len(alpha), ndatasets, len(initial_list)), np.nan)
negll = np.full((len(real_paras), len(alpha), ndatasets), np.nan)
AIC, BIC = np.full((len(real_paras), len(alpha), ndatasets), np.nan, float), np.full((len(real_paras), len(alpha), ndatasets), np.nan)


if generate_data == True:

    para_list = []

    bandit = BanditMoney()
    GenDesign.factor = 10 if use_10 else 1
    design = GenDesign()

    for p, para in enumerate(real_paras):

        locals()['m' + para] = realFitData.iloc[:, p].values.mean()
        para_list = np.append(para_list, eval('m' + para))

    for pa, paramet in enumerate(real_paras):

        loop_index = real_paras.index(paramet)
        parameter = para_list

        for l, loop in enumerate(eval(paramet.lower())):

            print("Simulating dataset for " + str(paramet) + " : trial " + str(l + 1) + " out of 100")

            parameter[loop_index] = loop

            sim_model = winning_model(*parameter)

            for i in np.arange(0, ndatasets):

                np.random.seed(i)
                design.generate()
                sim_model.noise = np.random.rand(1)

                for b in range(nblocks):

                    sim_model.values = np.full(nbandits, 0, float)

                    bandit.reset_outcome_history()
                    bandit.set_outcome_schedule(design.outcome_schedule[b], design.outcome_base[b], design.outcome_diff[b])

                    for p in range(nphases):
                        for t, tri in enumerate(np.where(~np.isnan(stim_left[i, b, p]))[0]):

                            sim_model.get_current_trial(tri)
                            sim_model.stims = np.array([int(stim_left[i, b, p, tri]), int(stim_right[i, b, p, tri])])

                            cp = sim_model.get_choice_probab()
                            choice[i, pa, l, b, p, t], choice_index = sim_model.simulated_choice()

                            sim_model.stim_chosen = int(choice[i, pa, l, b, p, t])

                            out = bandit.sample(int(choice[i, pa, l, b, p, t]), ignore_history_constraints=history_constraint[i, b, p, tri])
                            out_val[i, pa, l, b, p, t] = out if (p != 1) else np.nan

                            conf_val[i, pa, l, b, p, t] = sim_model.simulated_confidence(choice_index)

                            sim_model.update(out_val[i, pa, l, b, p, t], conf_val[i, pa, l, b, p, t])

sim_variables = ['choice', 'out_val', 'conf_val']

simu_data = ['para_choice', 'para_out', 'para_conf']
para_choice, para_out, para_conf = None, None, None

print('Load data')

for v, var in enumerate(sim_variables):

    print(f'v = {v + 1} / {len(sim_variables)}')

    if use_10:
        if (generate_data == True):
            np.save(os.path.join(path_save, f'{simu_data[v]}_10'), eval(var))
        else:
            locals()[simu_data[v]] = np.load(os.path.join(path_save, simu_data[v] + '_10.npy'))
    else:
        if (generate_data == True):
            np.save(os.path.join(path_save, f'{simu_data[v]}'), eval(var))
        else:
            locals()[simu_data[v]] = np.load(os.path.join(path_save, simu_data[v] + '.npy'))

print('End load data')

def recov_params(paras, running_model, s, simulation_model, return_cp=False):

    winModel = running_model(*paras)
    p_id = eval(simulation_model[0])
    l_id = eval(simulation_model[1])

    negLogL = 0

    if return_cp:
        choiceprob = np.full((nblocks, nphases, ntrials_phase_max), np.nan)

    for b in range(nblocks):

        winModel.values = np.full(nbandits, 0, float)

        for p in range(nphases):
            for t, tri in enumerate(np.where(~np.isnan(stim_left[s, b, p]))[0]):

                winModel.get_current_trial(tri)

                winModel.stims = np.array([int(stim_left[s, b, p, tri]), int(stim_right[s, b, p, tri])])
                winModel.stim_chosen = int(para_choice[s, p_id, l_id, b, p, t])
                cp = winModel.get_choice_probab()

                if return_cp:
                    choiceprob[b, p, t] = cp

                negLogL -= np.log(np.maximum(cp, 1e-8))

                winModel.update(para_out[s, p_id, l_id, b, p, t], para_conf[s, p_id, l_id, b, p, t])

    return (negLogL, choiceprob) if (return_cp == True) else negLogL


if __name__ == '__main__':

    para_list = []
    for p, para in enumerate(real_paras):
        locals()['m' + para] = realFitData.iloc[:, p].values.mean()
        para_list = np.append(para_list, eval('m' + para))

    for pa, paramet in enumerate(real_paras):

        print(f'pa = {pa + 1} / {len(real_paras)}')

        loop_index = real_paras.index(paramet)
        parameter = para_list

        for l, loop in enumerate(eval(paramet.lower())):

            if (paramet == 'BETA') and (np.isclose(loop, 0.7)):
            # if True:

                locals()['m' + para] = realFitData.iloc[:, pa].values.mean()
                para_list = np.append(para_list, eval('m' + para))
                parameter[loop_index] = loop
                print('Simulated parameters:', parameter)

                print("Simulating dataset for " + str(paramet) + " : trial " + str(l + 1) + " out of 100")

                for i in np.arange(0, ndatasets):

                    np.random.seed(i)

                    # pars = initial_list
                    simu_id = [str(pa), str(l)]
                    fitting.set_model(i, ndatasets, winning_model, recov_params, 4, simu_id)
                    fitting.local_minima(expect, bounds, grid_range, grid_multiproc=False)

                    for p in range(len(real_paras)):
                        parafit[pa, l, i, p] = min(bounds[p][1], max(bounds[p][0], fitting.data[i, p]))

                    negll[pa, l, i], probab_choice[pa, l, i] = recov_params(fitting.data[i], winning_model, i, simu_id, return_cp=True)
                    nsamples = np.sum(~np.isnan(probab_choice[pa, l, i]))

                    AIC[pa, l, i], BIC[pa, l, i] = fitting.model_fit(negll[pa, l, i], nsamples)

                    if i == (ndatasets - 1):

                        parameter_fit = pd.DataFrame(data={"ALPHA_" + str(pa) + '_' + str(l) + 'e': parafit[pa, l, :, 0],
                                                           "BETA_" + str(pa) + '_' + str(l) + 'e': parafit[pa, l, :, 1],
                                                           "ALPH_N_" + str(pa) + '_' + str(l) + 'e': parafit[pa, l, :, 2],
                                                           "GAMMA_" + str(pa) + '_' + str(l) + 'e': parafit[pa, l, :, 3]
                                                           },
                                                     columns=["ALPHA_" + str(pa) + '_' + str(l) + 'e',
                                                              "BETA_" + str(pa) + '_' + str(l) + 'e',
                                                              "ALPH_N_" + str(pa) + '_' + str(l) + 'e',
                                                              "GAMMA_" + str(pa) + '_' + str(l) + 'e'
                                                              ])
                        saveParameters = pd.concat([saveParameters, parameter_fit], axis=1)

                        model_fit = pd.DataFrame(data={"AIC_" + str(pa) + '_' + str(l) + 'e': AIC[pa, l, :],
                                                       "BIC_" + str(pa) + '_' + str(l) + 'e': BIC[pa, l, :],
                                                       "NEGLL_" + str(pa) + '_' + str(l) + 'e': negll[pa, l, :]
                                                       },
                                                 columns=["AIC_" + str(pa) + '_' + str(l) + 'e',
                                                          "BIC_" + str(pa) + '_' + str(l) + 'e',
                                                          "NEGLL_" + str(pa) + '_' + str(l) + 'e'
                                                          ])

                        saveFitting = pd.concat([saveFitting, model_fit], axis=1)

                        choice_probability = pd.DataFrame(data={"cp_" + str(pa) + '_' + str(l) + 'e': probab_choice[pa, l, :][~np.isnan(probab_choice[pa, l, :])]},
                                                          columns=["cp_" + str(pa) + '_' + str(l) + 'e'])

                        saveChoiceProbab = pd.concat([saveChoiceProbab, choice_probability], axis=1)

        if pa == (len(real_paras) - 1):

            if use_10:
                pd.concat([saveParameters, saveFitting], axis=1).to_pickle("fittingData_para_simu_10.pkl", protocol=4)
                saveChoiceProbab.to_pickle("choiceProbab_para_simu_10.pkl", protocol=4)
            else:
                pd.concat([saveParameters, saveFitting], axis=1).to_pickle("fittingData_para_simu.pkl", protocol=4)
                saveChoiceProbab.to_pickle("choiceProbab_para_simu.pkl", protocol=4)
