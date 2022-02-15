import numpy as np
import warnings
from .rl_simple import Rescorla

warnings.filterwarnings("ignore", category=RuntimeWarning)


class RescorlaChoiceDual(Rescorla):
    """model updates expected values according to confidence prediction error in all phases"""

    def __init__(self, alpha=0.1, beta=1, gamma=0.1, nbandits=5):
        """function introduces distinct learning parameters, gamma and alpha_c, for confidence-based updates"""

        super().__init__(alpha=alpha, beta=beta, nbandits=nbandits)

        self.gamma = gamma

    def update(self, outcome, confidence=None):

        if np.isnan(outcome):
            return self.choice_effect()
        else:
            self.choice_effect()
            return self.learn_value(outcome)

    def choice_effect(self):
        """confidence update operates in line with Rescorla Wagner learning rule"""

        self.values[self.stim_chosen] += self.gamma    # self.values[self.choice_predicted] += self.gamma

        return self.values[self.stim_chosen]


class RescorlaChoiceMono(RescorlaChoiceDual):
    """model implements confidence baseline, which tracks confidence updates in phase 0 and 2"""

    def __init__(self, alpha=0.1, beta=1, gamma=0.1, nbandits=5):

        super().__init__(alpha=alpha, beta=beta, gamma=gamma, nbandits=nbandits)

    def update(self, outcome, confidence=None):

        if np.isnan(outcome):
            return self.choice_effect()
        else:
            return self.learn_value(outcome)


class RescorlaChoiceDualDeval(RescorlaChoiceDual):
    """function updates learned values according to confidence PE and assumes an expected outcome of 0 in phase 1"""

    def __init__(self, alpha=0.1, beta=1, gamma=0.1, alpha_n=0.1, nbandits=5):

        super().__init__(alpha=alpha, beta=beta, gamma=gamma, nbandits=nbandits)

        self.alpha_n = alpha_n

    def update(self, outcome, confidence=None):

        if np.isnan(outcome):
            self.choice_effect()
            return self.learn_without_outcome()
        else:
            self.choice_effect()
            return self.learn_value(outcome)

    def learn_without_outcome(self):
        """function introduces new learning parameter alpha_n to capture the dynamics of learning in phase 1"""

        self.PE = 0 - self.values[self.stim_chosen]
        self.values[self.stim_chosen] += self.alpha_n * self.PE

        return self.values[self.stim_chosen]


class RescorlaChoiceMonoDeval(RescorlaChoiceMono):

    def __init__(self, alpha=0.1, beta=1, gamma=0.1, alpha_n=0.1, nbandits=5):

        super().__init__(alpha=alpha, beta=beta, gamma=gamma, nbandits=nbandits)

        self.alpha_n = alpha_n

    def update(self, outcome, confidence=None):

        if np.isnan(outcome):
            self.choice_effect()
            return self.learn_without_outcome()
        else:
            return self.learn_value(outcome)

    def learn_without_outcome(self):
        """function introduces new learning parameter alpha_n to capture the dynamics of learning in phase 1"""

        self.PE = 0 - self.values[self.stim_chosen]
        self.values[self.stim_chosen] += self.alpha_n * self.PE

        return self.values[self.stim_chosen]