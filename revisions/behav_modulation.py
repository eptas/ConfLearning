import os
import numpy as np
import pandas as pd

from pathlib import Path

from ConfLearning.models.rl_simple_simchoice import RescorlaConfBase, RescorlaConfBaseGen
from ConfLearning.models.rl_simple_choice_simchoice import RescorlaChoiceMono

from ConfLearning.models.maximum_likelihood import ParameterFit
from ConfLearning.recovery.bandit import BanditMoney
from ConfLearning.recovery.gen_design import GenDesign

cwd = Path.cwd()

genData = True
use_10 = True

fitting = ParameterFit()
bandit = BanditMoney()
GenDesign.factor = 10 if use_10 else 1
design = GenDesign()

modellist = [RescorlaChoiceMono, RescorlaConfBase, RescorlaConfBaseGen]
model_names = ['RescorlaChoiceMono', 'RescorlaConfBase', 'RescorlaConfBaseGen']

ndatasets = 100
nblocks = 11
nphases = 3
ntrials_phase_max = 57
nbandits = 5

design_vars = ['stim_left', 'stim_right', 'history_constraint', 'correct_value', 'pair', 'equal_value_pair', 'trial_phase', 'value_id']
stim_left, stim_right, history_constraint, correct_value, pair, equal_value_pair, trial_phase, value_id = None, None, None, None, None, None, None, None

print('Load design data')

for v, var in enumerate(design_vars):

    print(f'v = {v + 1} / {len(design_vars)}')

    if use_10:
        locals()[design_vars[v]] = np.load(os.path.join(cwd, design_vars[v] + '_mani_10.npy'))
    else:
        locals()[design_vars[v]] = np.load(os.path.join(cwd, design_vars[v] + '_mani.npy'))


alpha = 0.2
beta = 1/3
alpha_c = [0, 0.25, 0.5, 0.75, 1]
gamma = [0, 0.1, 1, 10, 100]


if genData == True:

    choice = np.full((ndatasets, len(alpha_c), len(gamma), nblocks, nphases, ntrials_phase_max), np.nan)
    out_val = np.full((ndatasets, len(alpha_c), len(gamma), nblocks, nphases, ntrials_phase_max), np.nan)
    conf_val = np.full((ndatasets, len(alpha_c), len(gamma), nblocks, nphases, ntrials_phase_max), np.nan)


def simulate_behav(model_id, sim_model):

    for alc, alph_c in enumerate(alpha_c):
        for gam, gam_a in enumerate(gamma):

            print("Simulating alpha_c " + str(alc + 1) + " out of 4 + gamma " + str(gam + 1) + " out of 4")

            sim_parameter = [alpha, beta, gam_a] if model_id == 0 else [alpha, beta, alph_c, gam_a]

            simModel = sim_model(*sim_parameter)

            for i in np.arange(0, ndatasets):

                print("Subject " + str(i + 1) + " out of " + str(ndatasets))

                np.random.seed(i)
                design.generate()
                simModel.noise = np.random.rand(1)

                for block in range(nblocks):

                    simModel.values = np.full(nbandits, 0, float)

                    bandit.reset_outcome_history()
                    bandit.set_outcome_schedule(design.outcome_schedule[block], design.outcome_base[block], design.outcome_diff[block])

                    for phase in range(nphases):
                        for tria, trials in enumerate(np.where(~np.isnan(eval("stim_left")[i, block, phase]))[0]):

                            simModel.get_current_trial(trials)
                            simModel.stims = np.array([int(eval("stim_left")[i, block, phase, trials]), int(eval("stim_right")[i, block, phase, trials])])

                            cp = simModel.get_choice_probab()

                            choice[i, alc, gam, block, phase, tria], choice_index = simModel.simulated_choice()
                            simModel.stim_chosen = int(choice[i, alc, gam, block, phase, tria])

                            out = bandit.sample(int(choice[i, alc, gam, block, phase, tria]), ignore_history_constraints=eval("history_constraint")[i, block, phase, trials])
                            out_val[i, alc, gam, block, phase, tria] = out if (phase != 1) else np.nan

                            conf_val[i, alc, gam, block, phase, tria] = simModel.simulated_confidence(choice_index)

                            simModel.update(out_val[i, alc, gam, block, phase, tria], conf_val[i, alc, gam, block, phase, tria])

    return choice, out_val, conf_val


for mo, model in enumerate(modellist):

    if genData == True:

        choices, outcomes, confidences = simulate_behav(mo, model)

        np.save('simu_choices_' + model_names[mo] + '.npy', choices)
        np.save('simu_outcomes_' + model_names[mo] + '.npy', outcomes)
        np.save('simu_confidence_' + model_names[mo] + '.npy', confidences)
