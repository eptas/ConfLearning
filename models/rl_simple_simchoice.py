import numpy as np
import warnings
from .rl_simple import Rescorla

warnings.filterwarnings("ignore", category=RuntimeWarning)



class RescorlaConf(Rescorla):
    """model updates expected values according to confidence prediction error in all phases"""

    def __init__(self, alpha=0.1, beta=1, alpha_c=0.1, gamma=0.1, nbandits=5):
        """function introduces distinct learning parameters, gamma and alpha_c, for confidence-based updates"""

        super().__init__(alpha=alpha, beta=beta, nbandits=nbandits)

        self.alpha_c = alpha_c
        self.gamma = gamma

        self.conf_values = np.full(self.nbandits, 0, float)
        self.conf_PE = None

    def update(self, outcome, confidence):

        if np.isnan(outcome):
            return self.learn_confidence_value(confidence)
        else:
            self.learn_confidence_value(confidence)

            return self.learn_value(outcome)

    def learn_confidence_value(self, confidence):
        """confidence update operates in line with Rescorla Wagner learning rule"""

        self.conf_PE = confidence - self.conf_values[self.stim_chosen]
        self.conf_values[self.stim_chosen] += self.alpha_c * self.conf_PE
        self.values[self.choice_predicted] += self.gamma * self.conf_PE

        return self.values[self.stim_chosen]


class RescorlaConfGen(RescorlaConf):
    """model uses generic (overall) confidence PE to update belief estimates in all phases"""

    def __init__(self, alpha=0.1, beta=1, alpha_c=0.1, gamma=0.1, nbandits=5):

        super().__init__(alpha=alpha, beta=beta, alpha_c=alpha_c, gamma=gamma, nbandits=nbandits)

        self.conf_values = 0

    def learn_confidence_value(self, confidence):
        """generic (overall) confidence estimate is used rather than distinct confidence values for each bandit"""

        self.conf_PE = confidence - self.conf_values
        self.conf_values += self.alpha_c * self.conf_PE
        self.values[self.choice_predicted] += self.gamma * self.conf_PE

        return self.values[self.choice_predicted]

    def get_confidence_exp_pe(self, confidence):

        conf_PE = confidence - self.conf_values    # self.conf_PE = confidence - self.conf_values
        conf_values_post = self.conf_values + self.alpha_c * conf_PE
        conf_values_pre = self.conf_values

        return conf_PE, conf_values_pre, conf_values_post


class RescorlaConfBase(RescorlaConf):
    """model implements confidence baseline, which tracks confidence updates in phase 0 and 2"""

    def __init__(self, alpha=0.1, beta=1, alpha_c=0.1, gamma=0.1, nbandits=5):

        super().__init__(alpha=alpha, beta=beta, alpha_c=alpha_c, gamma=gamma, nbandits=nbandits)

    def update(self, outcome, confidence):

        if np.isnan(outcome):
            return self.learn_confidence_value(confidence)
        else:
            self.track_confidence_value(confidence)

            return self.learn_value(outcome)

    def track_confidence_value(self, confidence):

        self.conf_PE = confidence - self.conf_values[self.stim_chosen]
        self.conf_values[self.stim_chosen] += self.alpha_c * self.conf_PE

    def get_confidence_exp_pe(self, confidence):

        conf_PE = confidence - self.conf_values[self.stim_chosen]    # self.conf_PE = confidence - self.conf_values
        conf_values_post = self.conf_values[self.stim_chosen] + self.alpha_c * conf_PE
        conf_values_pre = self.conf_values[self.stim_chosen]

        return conf_PE, conf_values_pre, conf_values_post


class RescorlaConfBaseGen(RescorlaConfGen):
    """model implments confidence baseline for generic (overall) confidence value"""

    def __init__(self, alpha=0.1, beta=1, alpha_c=0.1, gamma=0.1, nbandits=5):

        super().__init__(alpha=alpha, beta=beta, alpha_c=alpha_c, gamma=gamma, nbandits=nbandits)

    def update(self, outcome, confidence):

        if np.isnan(outcome):
            return self.learn_confidence_value(confidence)
        else:
            self.track_confidence_value(confidence)

            return self.learn_value(outcome)

    def track_confidence_value(self, confidence):

        self.conf_PE = confidence - self.conf_values
        self.conf_values += self.alpha_c * self.conf_PE


class RescorlaConfZero(RescorlaConf):
    """function updates learned values according to confidence PE and assumes an expected outcome of 0 in phase 1"""

    def __init__(self, alpha=0.1, beta=1, alpha_c=0.1, gamma=0.1, alpha_n=0.1, nbandits=5):

        super().__init__(alpha=alpha, beta=beta, alpha_c=alpha_c, gamma=gamma, nbandits=nbandits)

        self.alpha_n = alpha_n

    def update(self, outcome, confidence):

        if np.isnan(outcome):
            self.learn_confidence_value(confidence)

            return self.learn_without_outcome()
        else:
            self.learn_confidence_value(confidence)

            return self.learn_value(outcome)

    def learn_without_outcome(self):
        """function introduces new learning parameter alpha_n to capture the dynamics of learning in phase 1"""

        self.PE = 0 - self.values[self.stim_chosen]
        self.values[self.stim_chosen] += self.alpha_n * self.PE

        return self.values[self.stim_chosen]


class RescorlaConfZeroGen(RescorlaConfGen):
    """function updates learned values according to generic (overall) confidence PE with an expected outcome of 0 in phase 1"""

    def __init__(self, alpha=0.1, beta=1, alpha_c=0.1, gamma=0.1, alpha_n=0.1, nbandits=5):

        super().__init__(alpha=alpha, beta=beta, alpha_c=alpha_c, gamma=gamma, nbandits=nbandits)

        self.alpha_n = alpha_n

    def update(self, outcome, confidence):

        if np.isnan(outcome):
            self.learn_confidence_value(confidence)

            return self.learn_without_outcome()
        else:
            self.learn_confidence_value(confidence)

            return self.learn_value(outcome)

    def learn_without_outcome(self):
        """function introduces new learning parameter alpha_n to capture the dynamics of learning in phase 1"""

        self.PE = 0 - self.values[self.stim_chosen]
        self.values[self.stim_chosen] += self.alpha_n * self.PE

        return self.values[self.stim_chosen]


class RescorlaConfBaseZero(RescorlaConfBase):

    def __init__(self, alpha=0.1, beta=1, alpha_c=0.1, gamma=0.1, alpha_n=0.1, nbandits=5):

        super().__init__(alpha=alpha, beta=beta, alpha_c=alpha_c, gamma=gamma, nbandits=nbandits)

        self.alpha_n = alpha_n

    def update(self, outcome, confidence):

        if np.isnan(outcome):
            self.learn_confidence_value(confidence)

            return self.learn_without_outcome()
        else:
            self.track_confidence_value(confidence)

            return self.learn_value(outcome)

    def learn_without_outcome(self):
        """function introduces new learning parameter alpha_n to capture the dynamics of learning in phase 1"""

        self.PE = 0 - self.values[self.stim_chosen]
        self.values[self.stim_chosen] += self.alpha_n * self.PE

        return self.values[self.stim_chosen]


class RescorlaConfBaseZeroGen(RescorlaConfBaseGen):
    """Outcome of Zero"""

    def __init__(self, alpha=0.1, beta=1, alpha_c=0.1, gamma=0.1, alpha_n=0.1, nbandits=5):

        super().__init__(alpha=alpha, beta=beta, alpha_c=alpha_c, gamma=gamma, nbandits=nbandits)

        self.alpha_n = alpha_n

    def update(self, outcome, confidence):

        if np.isnan(outcome):
            self.learn_confidence_value(confidence)

            return self.learn_without_outcome()
        else:
            self.track_confidence_value(confidence)

            return self.learn_value(outcome)

    def learn_without_outcome(self):
        """function introduces new learning parameter alpha_n to capture the dynamics of learning in phase 1"""

        self.PE = 0 - self.values[self.stim_chosen]
        self.values[self.stim_chosen] += self.alpha_n * self.PE

        return self.values[self.stim_chosen]
