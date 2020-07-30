import numpy as np


class TwoArmedBandit:

    def __init__(self, nbandits=5, values=None, sd=1):

        self.nbandits = nbandits
        self.values = np.arange(nbandits) if values is None else np.array(values)
        self.sd = sd

    def get_stimuli(self):
        """this function outputs two random stimulus indices"""

        return np.random.randint(0, self.nbandits, 2)

    def get_outcome(self, choice):
        """this function outputs a reward value matrix for random stimulus presentation"""

        return np.random.normal(loc=self.values[choice], scale=self.sd)
