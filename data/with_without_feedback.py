import pandas as pd
import numpy as np

data_path = 'C:/Users/esthe/Desktop/data.pkl'


class ExtractData:

    def __init__(self):
        """This function loads the pandas DataFrame from pickle file for subsequent data extraction."""

        self.data = pd.read_pickle(data_path, compression=None)
        self.df = None

    def set_trial(self, s, b, p, t):
        """This function sets current subject, phase, block and trial for sequential analysis."""

        self.df = self.data[(self.data.subject == s) & (self.data.block == b) & (self.data.phase == p) & (self.data.trial == t)]

    def get_stimuli(self):
        """Function outputting the index of the 2AFC bandits (out of 5)."""

        stim1 = self.df.stimulus_left.values[0]
        stim2 = self.df.stimulus_right.values[0]
        stims = np.array([stim1, stim2])

        return stims

    def get_choice(self):
        """This function extracts the index of the participant's choice."""

        stim_chosen = self.df.choice.values[0]

        return stim_chosen

    def get_outcome(self):
        """This function simulates the reward feedback."""

        outcome = self.df.outcome.values[0]

        return outcome

    def get_confidence(self):
        """This function extracts the confidence ratings for each trial."""

        confidence = self.df.confidence.values[0]

        return confidence

    def get_correct(self):

        correct = self.df.type_choice.values[0]

        return correct

    def get_true_values(self, s, b, nbandits):

        true_value = [self.data[(self.data.subject == s) & (self.data.block == b) & (self.data.type_rating3) & (self.data.stimulus_left == i)].value_chosen.values[0] for i in range(nbandits)]

        return true_value
