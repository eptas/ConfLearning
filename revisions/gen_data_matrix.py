import os
import numpy as np
import pandas as pd

from ConfLearning.recovery.gen_design import GenDesign
from pathlib import Path

gen_data = True
use_10 = False
ndatasets = 100

# dfs = [None] * ndatasets
exclusion_list = ['trial_running', 'trial_sync', 'stimulus_left_outcome_id', 'stimulus_right_outcome_id', 'stimulus_id_left', 'stimulus_id_right', 'ntrials_noc', 'type_choice_noc']

if not os.path.exists('para_experiment'):
    os.makedirs('para_experiment')

path_save = os.path.join(Path.cwd(), './para_experiment/')

if gen_data == True:

    GenDesign.factor = 10 if use_10 else 1
    genDesign = GenDesign()

    for n in np.arange(ndatasets):
        print(f'n {n + 1} / {ndatasets}')

        np.random.seed(n)
        experiment = genDesign.generate()
        experiment.drop(columns=exclusion_list)
        # dfs[n] = experiment

        if use_10:
            experiment.to_pickle(os.path.join(path_save, "exp_mani_" + str(n) + "_10.pkl"), protocol=4)
        else:
            experiment.to_pickle(os.path.join(path_save, "exp_mani_" + str(n) + ".pkl"), protocol=4)

    # df_combined = pd.concat(dfs).reset_index(drop=True)
    # df_combined.to_pickle('exp_mani_matrix.pkl', protocol=4)
