#!/fast/home/users/ptasczle_c/python/bin/python3 -u
from concurrent.futures import ProcessPoolExecutor as Pool
import os
import numpy as np
import pandas as pd
from timeit import default_timer
from itertools import product as prod
from pathlib import Path
import sys
import pickle
HOME = os.path.expanduser("~")
sys.path.extend([os.path.join(HOME, 'work/Dropbox/confidence/')])
from ConfLearning.models.rl_simple_simchoice import RescorlaConfBase, RescorlaConfBaseGen
from ConfLearning.models.rl_simple import RescorlaPerseveration
from ConfLearning.models.rl_simple_choice_simchoice import RescorlaChoiceMono

from ConfLearning.revision2.bandit import BanditMoney
from ConfLearning.revision2.gen_design_sim import GenDesign
from functools import partial


use_10 = False

bandit = BanditMoney()
GenDesign.factor = 10 if use_10 else 1
design = GenDesign()

modellist = [RescorlaChoiceMono, RescorlaConfBase, RescorlaConfBaseGen, RescorlaPerseveration]
model_names = ['ChoiceMono', 'ConfBase', 'ConfBaseGen', 'Perseveration']

ndatasets = 100
# ndatasets = 2
nblocks = 11
nphases = 3
nbandits = 5

if os.path.exists(os.path.join(HOME, 'work/Dropbox/confidence/ConfLearning/revisions/para_experiment/')):
    design_path = os.path.join(HOME, 'work/Dropbox/confidence/ConfLearning/revisions/para_experiment/')
else:
    design_path = os.path.join(Path.cwd(), '../plot/data/para_experiment/')

alpha = 0.2
beta = 0.2
# alpha_c = [0, 0.25, 0.5, 0.75, 1]
# gamma = [0, 0.1, 1, 10, 100]
alpha_c = np.linspace(0, 1, 3)
# gamma = np.exp(np.linspace(np.log(1), np.log(50), 8))
gamma = np.exp(np.linspace(0, 4, 8))
lambd = np.hstack((0, np.exp(np.linspace(np.log(0.5), np.log(10), 7))))
eta = np.hstack((np.linspace(-1.5, 1.5, 7), np.nan))

# alpha_c = [0, 0.25]
# gamma = [0, 0.1]


def loop(bulkid, i):

    t0 = default_timer()

    alc, gam = combos[i]

    model_frames = [None] * len(modellist)
    for mo, model in enumerate(modellist):

        if model_names[mo] == 'ChoiceMono':
            alph_c, gam_a = alpha_c[alc], lambd[gam]
        elif model_names[mo] == 'Perseveration':
            alph_c, gam_a = alpha_c[alc], eta[gam]
        else:
            alph_c, gam_a = alpha_c[alc], gamma[gam]

        if np.isnan(gam_a):
            model_frames[mo] = pd.DataFrame()
        else:
            t1 = default_timer()

            # print("Simulating alpha_c " + str(alc + 1) + " out of 5 + gamma " + str(gam + 1) + " out of 5")

            sim_parameter = [alpha, beta, gam_a] if mo in (0, 3) else [alpha, beta, alph_c, gam_a]
            simModel = model(*sim_parameter)

            param_frame = [None] * ndatasets

            for n in np.arange(0, ndatasets):

                # print("Subject " + str(n + 1) + " out of " + str(ndatasets))
                t2 = default_timer()

                np.random.seed(n+bulkid*ndatasets)
                design.generate()

                # print(os.path.join(design_path, 'exp_mani_' + str(n) + '.pkl'))
                sub_frame = pd.read_pickle(os.path.join(design_path, 'exp_mani_' + str(n) + '.pkl'))
                sub_frame['model_id'] = mo
                sub_frame['model'] = model_names[mo]
                sub_frame['alpha_c_id'] = alc
                sub_frame['alpha_c'] = alph_c
                sub_frame['gamma_id'] = gam
                sub_frame['gamma'] = gam_a
                sub_frame['subject'] = n

                for block in range(nblocks):

                    simModel.values = np.full(nbandits, 0, float)

                    bandit.reset_outcome_history()
                    bandit.set_outcome_schedule(design.outcome_schedule[block], design.outcome_base[block], design.outcome_diff[block])

                    equal_value_stims = np.array(sorted(np.unique(np.hstack((
                        sub_frame[(sub_frame.block == block) & sub_frame.equal_value_pair].stimulus_left,
                        sub_frame[(sub_frame.block == block) & sub_frame.equal_value_pair].stimulus_right
                    )))), dtype=int)
                    value_order = np.array([v-(v>=equal_value_stims.max()) for v in range(5)])

                    for phase in range(nphases):
                        for tria, trials in enumerate(sub_frame[(sub_frame.block == block) & (sub_frame[~sub_frame.phase.isna()].phase == phase) & sub_frame.type_choice_obs].trial_phase.astype(int).values):

                            simModel.noise = np.random.rand(1)

                            curr_data = sub_frame[(sub_frame.block == block) & (sub_frame.phase == phase) & (sub_frame.trial_phase == trials)]
                            simModel.get_current_trial(trials)
                            simModel.stims = np.array([int(curr_data.stimulus_left.item()), int(curr_data.stimulus_right.item())])

                            simModel.get_choice_probab()

                            choice, choice_index = simModel.simulated_choice()
                            simModel.stim_chosen = int(choice)
                            unchosen = int(simModel.stims[0]) if int(simModel.stims[1]) == int(choice) else int(simModel.stims[1])

                            out = bandit.sample(int(choice), ignore_history_constraints=curr_data.pre_equalshown_secondlasttrial.values[0])
                            out_val = out if (phase != 1) else np.nan

                            conf_val = simModel.simulated_confidence(choice_index)

                            cond = (sub_frame.block == block) & (sub_frame.phase == phase) & (sub_frame.trial_phase == trials)
                            sub_frame.loc[cond, 'choice'] = int(choice)
                            sub_frame.loc[cond, 'unchosen'] = unchosen
                            sub_frame.loc[cond, 'correct'] = value_order[int(choice)] >= value_order[unchosen]
                            sub_frame.loc[cond, 'value_order'] = value_order[int(choice)]
                            sub_frame.loc[cond, 'confidence'] = conf_val
                            sub_frame.loc[cond, 'outcome'] = out_val

                            for k in range(4):
                                if k == value_order[int(choice)]:
                                    sub_frame.loc[cond, f'confidence{k}'] = conf_val
                                else:
                                    if (phase > 0) & (trials == 0):
                                        if np.all(np.isnan(sub_frame[(sub_frame.block == block) & (sub_frame.phase == phase - 1)][f'confidence{k}'].values)):
                                            sub_frame.loc[cond, f'confidence{k}'] = sub_frame[(sub_frame.block == block) & (sub_frame.phase == phase - 2)][f'confidence{k}'].values[-1]
                                        else:
                                            sub_frame.loc[cond, f'confidence{k}'] = sub_frame[(sub_frame.block == block) & (sub_frame.phase == phase - 1)][f'confidence{k}'].values[-1]
                                    else:
                                        sub_frame.loc[cond, f'confidence{k}'] = 0 if trials == 0 else sub_frame[(sub_frame.block == block) & (sub_frame.phase == phase) & (sub_frame.trial_phase == trials - 1)][f'confidence{k}'].values[0]


                            simModel.update(out_val, conf_val)

                param_frame[n] = sub_frame
                # print(f'\t[{mo + 1} / {len(modellist)}] Inner loop {n + 1} / {ndatasets}: {default_timer() - t2:.1f} secs')
            print(f'\t[{i + 1} / {ncombos}] Inner loop {mo + 1} / {len(modellist)}: {default_timer() - t1:.1f} secs')

            model_frame = pd.concat(param_frame).reset_index(drop=True)
            cols = ['model_id', 'model', 'alpha_c_id', 'alpha_c', 'gamma_id', 'gamma', 'subject']
            model_frame = model_frame[cols + [c for c in model_frame.columns if c not in cols]]
            model_frames[mo] = model_frame

    print(f'Finished combo {i + 1} / {ncombos}: {default_timer() - t0:.1f} secs')
    return pd.concat(model_frames).reset_index(drop=True)


if __name__ == '__main__':

    if len(sys.argv) > 1:
        bulk_id = int(sys.argv[1])
    else:
        bulk_id = 0

    combos = list(prod(range(len(alpha_c)), range(len(gamma))))
    ncombos = len(combos)

    with Pool(24) as pool:
        result = list(pool.map(partial(loop, bulk_id), range(ncombos)))
    # result = [None] * ncombos
    # for i in range(ncombos):
    #     result[i] = loop(bulk_id, i)

    df = pd.concat(result).reset_index(drop=True)
    path_ = os.path.join(HOME, f'work/Dropbox/confidence/ConfLearning/results/behav_modulation_{bulk_id}.pkl.gz')
    df.to_pickle(path_, protocol=4)
    print(f'Saved to path {path_}')