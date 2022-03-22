import os
import numpy as np
import pandas as pd

from ConfLearning.recovery.gen_design import GenDesign
from pathlib import Path

gen_data = True
use_10 = True
ndatasets = 100

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

        if use_10:
            experiment.to_pickle(os.path.join(path_save, "experimental_sim_" + str(n) + "_10.pkl"), protocol=4)
        else:
            experiment.to_pickle(os.path.join(path_save, "experimental_sim_" + str(n) + ".pkl"), protocol=4)


nblocks = 110 if use_10 else 11
nphases = 3
ntrials = 18
nbandits = 5

stim_left = np.full((ndatasets, nblocks, nphases, ntrials), np.nan, float)
stim_right = np.full((ndatasets, nblocks, nphases, ntrials), np.nan, float)

history_constraint = np.full((ndatasets, nblocks, nphases, ntrials), np.nan, bool)
correct_value = np.full((ndatasets, nblocks, nphases, ntrials), np.nan, float)


class ExtractData:

    def __init__(self):

        self.data = None
        self.df = None

        self.stim_l = np.full((nblocks, nphases, ntrials), np.nan, float)
        self.stim_r = np.full((nblocks, nphases, ntrials), np.nan, float)

        self.history_con = np.full((nblocks, nphases, ntrials), np.nan, bool)
        self.correct_val = np.full((nblocks, nphases, ntrials), np.nan, float)

    def extract_data(self, dataset):
        """loops through dataframe to extract relevant arrays for subsequent fitting"""

        self.data = dataset

        for one, b in enumerate(self.data.block.unique()):
            for two, p in enumerate(self.data[(self.data.block == b) & (~self.data[~self.data.phase.isna()].phase.isna())].phase.unique()):
                    for three, t in enumerate(self.data[(self.data.block == b) & (self.data[~self.data.phase.isna()].phase == p) & self.data.type_choice_obs].trial_phase.astype(int).values):

                        self.df = self.data[(self.data.block == b) & (self.data.phase == p) & (self.data.trial_phase == t)]

                        self.stim_l[b, p, t] = self.df.stimulus_left.values[0]
                        self.stim_r[b, p, t] = self.df.stimulus_right.values[0]

                        self.history_con[b, p, t] = self.df.pre_equalshown_secondlasttrial.values[0]
                        self.correct_val[b, p, t] = self.df.type_choice.values[0]

        return self.stim_l, self.stim_r, self.history_con, self.correct_val


if __name__ == '__main__':

    data = ExtractData()

    for i in np.arange(0, ndatasets):

        print(f'i {i + 1} / {ndatasets}')

        if use_10:
            matrix = pd.read_pickle(os.path.join(path_save, 'experimental_sim_' + str(i) + '_10.pkl'))
        else:
            matrix = pd.read_pickle(os.path.join(path_save, 'experimental_sim_' + str(i) + '.pkl'))

        stim_left[i, :], stim_right[i, :], history_constraint[i, :], correct_value[i, :] = data.extract_data(matrix)

    variables = ['stim_left', 'stim_right', 'history_constraint', 'correct_value']

    for v, var in enumerate(variables):
        if use_10:
            np.save(os.path.join(path_save, f'{var}_10'), eval(var))
        else:
            np.save(os.path.join(path_save, f'{var}'), eval(var))
