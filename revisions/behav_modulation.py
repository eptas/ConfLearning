import os
import numpy as np
import pandas as pd

from pathlib import Path

from ConfLearning.models.rl_simple_simchoice import RescorlaConfBase, RescorlaConfBaseGen
from ConfLearning.models.rl_simple_choice_simchoice import RescorlaChoiceMono

from ConfLearning.recovery.bandit import BanditMoney
from ConfLearning.recovery.gen_design import GenDesign

cwd = Path.cwd()

use_10 = False

bandit = BanditMoney()
GenDesign.factor = 10 if use_10 else 1
design = GenDesign()

modellist = [RescorlaChoiceMono, RescorlaConfBase, RescorlaConfBaseGen]
model_names = ['ChoiceMono', 'ConfBase', 'ConfBaseGen']

ndatasets = 100
nblocks = 11
nphases = 3
nbandits = 5

design_path = os.path.join(Path.cwd(), './para_experiment/')

alpha = 0.2
beta = 1/3
alpha_c = [0, 0.25, 0.5, 0.75, 1]
gamma = [0, 0.1, 1, 10, 100]


for mo, model in enumerate(modellist):

    print(model_names[mo])
    for alc, alph_c in enumerate(alpha_c):
        for gam, gam_a in enumerate(gamma):

            print("Simulating alpha_c " + str(alc + 1) + " out of 5 + gamma " + str(gam + 1) + " out of 5")

            sim_parameter = [alpha, beta, gam_a] if mo == 0 else [alpha, beta, alph_c, gam_a]
            simModel = model(*sim_parameter)

            param_frame = [None] * ndatasets

            for i in np.arange(0, ndatasets):

                print("Subject " + str(i + 1) + " out of " + str(ndatasets))

                np.random.seed(i)
                design.generate()
                simModel.noise = np.random.rand(1)

                subject_frame = [None] * ndatasets

                for d in np.arange(0, ndatasets):

                    print("Dataset " + str(d))

                    sub_frame = pd.read_pickle(os.path.join(design_path, 'exp_mani_' + str(d) + '.pkl'))

                    for block in range(nblocks):

                        simModel.values = np.full(nbandits, 0, float)

                        bandit.reset_outcome_history()
                        bandit.set_outcome_schedule(design.outcome_schedule[block], design.outcome_base[block], design.outcome_diff[block])

                        for phase in range(nphases):
                            for tria, trials in enumerate(sub_frame[(sub_frame.block == block) & (sub_frame[~sub_frame.phase.isna()].phase == phase) & sub_frame.type_choice_obs].trial_phase.astype(int).values):

                                curr_data = sub_frame[(sub_frame.block == block) & (sub_frame.phase == phase) & (sub_frame.trial_phase == trials)]
                                simModel.get_current_trial(trials)
                                simModel.stims = np.array([int(curr_data.stimulus_left.values[0]), int(curr_data.stimulus_right.values[0])])

                                cp = simModel.get_choice_probab()

                                choice, choice_index = simModel.simulated_choice()
                                simModel.stim_chosen = int(choice)

                                out = bandit.sample(int(choice), ignore_history_constraints=curr_data.pre_equalshown_secondlasttrial.values[0])
                                out_val = out if (phase != 1) else np.nan

                                conf_val = simModel.simulated_confidence(choice_index)

                                sub_frame.loc[(sub_frame.block == block) & (sub_frame.phase == phase) & (sub_frame.trial_phase == trials), 'subjects'] = int(i)
                                sub_frame.loc[(sub_frame.block == block) & (sub_frame.phase == phase) & (sub_frame.trial_phase == trials), 'choice'] = int(choice)
                                sub_frame.loc[(sub_frame.block == block) & (sub_frame.phase == phase) & (sub_frame.trial_phase == trials), 'confidence'] = conf_val
                                sub_frame.loc[(sub_frame.block == block) & (sub_frame.phase == phase) & (sub_frame.trial_phase == trials), 'outcome'] = out_val

                                simModel.update(out_val, conf_val)

                    subject_frame[d] = sub_frame

                param_frame[i] = pd.concat(subject_frame).reset_index(drop=True)

            final_frame = pd.concat(param_frame).reset_index(drop=True)
            final_frame.to_pickle('sim_' + model_names[mo] + '_a' + str(alc) + '_g' + str(gam) + '.pkl', protocol=4)
