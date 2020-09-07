import os
import numpy as np
import pandas as pd

from pathlib import Path

cwd = Path.cwd()
path_data = os.path.join(cwd, '../data/')

matrix = pd.read_pickle(os.path.join(path_data, 'data.pkl'))

nsubjects = max(matrix.subject.values) + 1
nblocks = max(matrix.block.values) + 1
nphases = max(matrix[~matrix.phase.isna()].phase.values) + 1
ntrials = 56
nbandits = 5


class ExtractData:

    def __init__(self):

        self.data = matrix
        self.df = None

        self.stim_left = np.full((nsubjects, nblocks, nphases, ntrials), np.nan, float)
        self.stim_right = np.full((nsubjects, nblocks, nphases, ntrials), np.nan, float)

        self.chosen_stim = np.full((nsubjects, nblocks, nphases, ntrials), np.nan, float)
        self.correct_value = np.full((nsubjects, nblocks, nphases, ntrials), np.nan, float)

        self.outcome_value = np.full((nsubjects, nblocks, nphases, ntrials), np.nan, float)
        self.confidence_value = np.full((nsubjects, nblocks, nphases, ntrials), np.nan, float)

        self.true_value = np.full((nsubjects, nblocks, nbandits), np.nan, float)

    def extract_data(self):
        """loops through dataframe to extract relevant arrays for subsequent fitting"""

        for one, s in enumerate(self.data.subject.unique()):
            for two, b in enumerate(self.data[(self.data.subject == s)].block.unique()):
                for three, p in enumerate(self.data[(self.data.subject == s) & (self.data.block == b) & (
                        ~self.data[~self.data.phase.isna()].phase.isna())].phase.unique()):
                    for four, t in enumerate(self.data[(self.data.subject == s) & (self.data.block == b) & (
                            self.data[~self.data.phase.isna()].phase == p) & self.data.type_choice_obs].trial.values):

                        self.df = self.data[
                            (self.data.subject == s) & (self.data.block == b) & (self.data.phase == p) & (
                                    self.data.trial == t)]

                        self.stim_left[s, b, p, t] = self.df.stimulus_left.values[0]
                        self.stim_right[s, b, p, t] = self.df.stimulus_right.values[0]

                        self.chosen_stim[s, b, p, t] = self.df.choice.values[0]

                        self.outcome_value[s, b, p, t] = pd.to_numeric(self.df.outcome.values[0], errors='coerce')
                        self.confidence_value[s, b, p, t] = self.df.confidence.values[0]

                        self.correct_value[s, b, p, t] = self.df.type_choice.values[0]
                        self.true_value[s, b, :] = [self.data[(self.data.subject == s) & (self.data.block == b) & (
                            self.data.type_rating3) & (self.data.stimulus_left == i)].value_chosen.values[0] for i in
                                                    range(nbandits)]

        return self.stim_left, self.stim_right, self.chosen_stim, self.outcome_value, self.confidence_value, self.correct_value, self.true_value


if __name__ == '__main__':

    data = ExtractData()
    stim_left, stim_right, chosen_stim, outcome_value, confidence_value, correct_value, true_value = data.extract_data()

    variables = ['stim_left', 'stim_right', 'chosen_stim', 'outcome_value', 'confidence_value', 'correct_value', 'true_value']

    for v, var in enumerate(variables):
        np.save(var, eval(var))
