import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class Rescorla:
    """simple reinforcement learning model using the Rescorla Wagner learning rule"""

    def __init__(self, alpha=0.1, beta=1, nbandits=5):

        self.alpha = alpha
        self.beta = beta
        self.nbandits = nbandits

        self.stims = None
        self.stim_chosen = None
        self.stim_unchosen = None

        self.trial = None

        self.choice_probab = None
        self.PE = None

        self.confidence = None

        self.noise = None

        self.values = np.full(self.nbandits, 0, float)
        self.value_history = [[] for _ in range(self.nbandits)]

    def get_current_trial(self, trial):

        self.trial = trial + 1

    def get_choice_probab(self, outcome=None):
        """function outputs choice probability for chosen stimulus"""

        self.choice_probab = 1 / (1 + np.exp(-self.beta * (self.values[self.stims[1]] - self.values[self.stims[0]])))

        return self.choice_probab if self.stim_chosen == self.stims[1] else 1 - self.choice_probab

    def get_choice_probab2(self):
        """function outputs choice probability"""

        return 1 / (1 + np.exp(-self.beta * (self.values[self.stims[1]] - self.values[self.stims[0]])))

    def choice(self):
        """function simulates choice based on choice probability of left stimulus"""

        self.stim_chosen = self.stims[int(np.random.rand() < self.choice_probab)]
        self.stim_unchosen = self.stims[0] if self.stim_chosen == self.stims[1] else self.stims[1]

        return self.stim_chosen

    def predicted_choice(self):
        """function simulates choice based on choice probability of left stimulus"""

        self.choice_predict = self.stims[int(self.choice_probab >= 0.5)]

        return self.choice_predict

    def simulated_choice(self):

        self.choice_predicted, self.choice_simulated = self.stims[int(self.noise < self.choice_probab)], int(self.noise < self.choice_probab)

        return self.choice_predicted, self.choice_simulated

    def update(self, outcome, confidence=None):

        return self.learn_value(outcome) if ~np.isnan(outcome) else self.values[self.stim_chosen]

    def learn_value(self, outcome):
        """function updates learned bandit values according to Rescorla Wagner learning rule"""

        self.PE = outcome - self.values[self.stim_chosen]
        self.values[self.stim_chosen] += self.alpha * self.PE

        return self.values[self.stim_chosen]

    def learn_history(self):
        """function stores learning history of all bandit values"""

        self.value_history[self.stim_chosen] = self.value_history[self.stim_chosen] + [self.values[self.stim_chosen]]

        return self.value_history

    def get_confidence(self):
        """function simulates confidence ratings based on choice probability of left stimulus"""

        self.confidence = (np.abs(self.choice_probab - (self.choice_probab < 0.5)) - 0.5) * 2

        return self.confidence

    def simulated_confidence(self, choice_ind):

        choice_proba = self.choice_probab if choice_ind == 1 else 1 - self.choice_probab    # index either 0 or 1

        self.sim_confidence = (np.abs(choice_proba - (choice_proba < 0.5)) - 0.5) * 2

        return self.sim_confidence

    def get_confidence2(self):
        """function simulates confidence ratings based on choice probability of left stimulus"""

        self.confidence = 0 if self.choice_probab < 0.5 else 2 * (self.choice_probab - 0.5)

        return self.confidence

    def get_confidence_inv(self):
        """function simulates confidence ratings based on choice probability of left stimulus"""

        return 1 - self.get_confidence()


class RescorlaZero(Rescorla):
    """model learns in phase 1 based on outcome value of zero (no feedback)"""

    def __init__(self, alpha=0.1, beta=1, alpha_n=0.1, nbandits=5):

        super().__init__(alpha=alpha, beta=beta, nbandits=nbandits)

        self.alpha_n = alpha_n

    def update(self, outcome, confidence=None):

        if np.isnan(outcome):
            return self.learn_without_outcome()
        else:
            return self.learn_value(outcome)

    def learn_without_outcome(self):
        """function introduces new learning parameter alpha_n to capture the dynamics of learning in phase 1"""

        self.PE = 0 - self.values[self.stim_chosen]
        self.values[self.stim_chosen] += self.alpha_n * self.PE

        return self.values[self.stim_chosen]

# class RescorlaPreservation(Rescorla):
#     """model learns in phase 1 based on outcome value of zero (no feedback)"""
#
#     def __init__(self, alpha=0.1, beta=1, eta=0.1, nbandits=5):
#
#         super().__init__(alpha=alpha, beta=beta, nbandits=nbandits)
#
#         self.eta = eta
#
#         self.last_choices = []
#         # grid = np.arange(-0.4, 0.11, 0.2)
#         # lb = -1
#         # ub = 1
#
#     def get_choice_probab(self, outcome=None):
#         """function outputs choice probability for chosen stimulus"""
#
#         if np.isnan(outcome) and (self.stims[0] in self.last_choices or self.stims[1] in self.last_choices):
#             last_stim0 = np.where(np.array(self.last_choices) == self.stims[0])[0][-1] if self.stims[0] in self.last_choices else -1
#             last_stim1 = np.where(np.array(self.last_choices) == self.stims[1])[0][-1] if self.stims[1] in self.last_choices else -1
#             pres = int(last_stim1 > last_stim0)
#         else:
#             pres = 0
#         self.choice_probab = 1 / (1 + np.exp(-(self.beta * (self.values[self.stims[1]] - self.values[self.stims[0]]) + self.eta * pres)))
#
#         return self.choice_probab if self.stim_chosen == self.stims[1] else 1 - self.choice_probab
#
#     def update(self, outcome, confidence=None):
#
#         self.last_choices += [self.stim_chosen]
#
#
#         if np.isnan(outcome):
#             return self.values[self.stim_chosen]
#         else:
#             return self.learn_value(outcome)


class RescorlaPerservation(Rescorla):

    def __init__(self, alpha=0.1, beta=1, eta=0.1, nbandits=5):

        super().__init__(alpha=alpha, beta=beta, nbandits=nbandits)

        self.eta = eta

        self.last_choice = dict()

    def get_choice_probab(self, outcome=None):
        """function outputs choice probability for chosen stimulus"""

        if np.isnan(outcome) and tuple(sorted(self.stims)) in self.last_choice:
            pres = 1 if self.last_choice[tuple(sorted(self.stims))] == self.stims[1] else -1
        else:
            pres = 0
        self.choice_probab = 1 / (1 + np.exp(-(self.beta * (self.values[self.stims[1]] - self.values[self.stims[0]]) + self.eta * pres)))

        return self.choice_probab if self.stim_chosen == self.stims[1] else 1 - self.choice_probab

    def get_confidence(self):
        """function simulates confidence ratings based on choice probability of left stimulus"""

        choice_probab_ = 1 / (1 + np.exp(-self.beta * (self.values[self.stims[1]] - self.values[self.stims[0]])))

        self.confidence = (np.abs(choice_probab_ - (choice_probab_ < 0.5)) - 0.5) * 2

        return self.confidence

    def update(self, outcome, confidence=None):

        self.last_choice[tuple(sorted(self.stims))] = self.stim_chosen

        if np.isnan(outcome):
            return self.values[self.stim_chosen]
        else:
            return self.learn_value(outcome)

class RescorlaBetaSlope(Rescorla):
    """model learns in phase 1 based on outcome value of zero (no feedback)"""

    def __init__(self, alpha=0.1, beta=1, beta_slope=0.1, nbandits=5):

        super().__init__(alpha=alpha, beta=beta, nbandits=nbandits)

        self.beta_slope = beta_slope
        # grid = np.arange(-0.4, 0.11, 0.2)
        # lb = -1
        # ub = 1


    def update(self, outcome, confidence=None):

        if np.isnan(outcome):
            self.beta += self.beta_slope
            self.beta = np.maximum(1e-5, self.beta)
            return self.values[self.stim_chosen]
        else:
            return self.learn_value(outcome)


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
        self.values[self.stim_chosen] += self.gamma * self.conf_PE

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
        self.values[self.stim_chosen] += self.gamma * self.conf_PE

        return self.values[self.stim_chosen]

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


class BayesModel(Rescorla):

    def __init__(self, alpha=0.1, beta=1, alpha_c=3, phi=0.1, nsamples=None):

        super().__init__(alpha=alpha, beta=beta)

        self.alpha_c = alpha_c
        self.phi = phi

        self.nsamples = nsamples
        self.values_sigma = np.full(self.nbandits, 1, float)

    def get_choice_probab(self):

        delta_values = self.values[self.stims[0]] - self.values[self.stims[1]]
        delta_values_sigma = np.sqrt(self.values_sigma[self.stims[0]]) - np.sqrt(self.values_sigma[self.stims[1]])

        self.choice_probab = 1 / (1 + np.exp(-self.beta * (delta_values + self.phi * delta_values_sigma)))

        return self.choice_probab if self.stim_chosen == self.stims[1] else 1 - self.choice_probab

    def update(self, outcome, confidence):

        self.PE = outcome - self.values[self.stim_chosen]
        self.values_sigma[self.stim_chosen] += self.alpha * (self.PE ** 2 - self.values_sigma[self.stim_chosen])
        self.values[self.stim_chosen] += self.alpha * self.PE

        return self.values[self.stim_chosen]

    def get_confidence(self):

        pooled_sd = np.sqrt(np.sum(self.values_sigma))
        smd = (self.values[int(self.stim_chosen == 1)] - self.values[int(self.stim_chosen == 0)]) / pooled_sd

        self.confidence = (1 / (1 + np.exp(-self.alpha_c * smd)) - 0.5) * 2

        return self.confidence


class BayesIdealObserver(BayesModel):

    def __init__(self, alpha=0.1, beta=1, alpha_c=3, phi=0.1):

        super().__init__(alpha=alpha, beta=beta, alpha_c=alpha_c, phi=phi)

    def update(self, outcome, confidence):

        self.PE = outcome - self.values[self.stim_chosen]
        self.values_sigma[self.stim_chosen] = (1 - 1 / self.trial) * self.values_sigma[self.stim_chosen] + self.PE ** 2 / (1 + self.trial)
        # self.values_sigma[self.stim_chosen] = self.values_sigma[self.stim_chosen] + (self.PE ** 2/( 1 + self.trial) - self.values_sigma[self.stim_chosen] / self.trial)  # equivalent
        self.values[self.stim_chosen] = (self.trial * self.values[self.stim_chosen] + outcome) / (self.trial + 1)

        return self.values[self.stim_chosen]


class RescorlaConfGamma(RescorlaConf):

    def __init__(self, alpha=0.1, beta=1, alpha_c_f=0.1, alpha_c_wo=0.1, gamma=0.1, nbandits=5):

        super().__init__(alpha=alpha, beta=beta, gamma=gamma, nbandits=nbandits)

        self.alpha_c_f = alpha_c_f
        self.alpha_c_wo = alpha_c_wo

    def update(self, outcome, confidence):

        if np.isnan(outcome):
            return self.learn_confidence_value_without_feed(confidence)
        else:
            self.learn_confidence_value_with_feed(confidence)

            return self.learn_value(outcome)

    def learn_confidence_value_with_feed(self, confidence):

        self.conf_PE = confidence - self.conf_values[self.stim_chosen]
        self.conf_values[self.stim_chosen] += self.alpha_c_f * self.conf_PE
        self.values[self.stim_chosen] += self.gamma * self.conf_PE

        return self.values[self.stim_chosen]

    def learn_confidence_value_without_feed(self, confidence):

        self.conf_PE = confidence - self.conf_values[self.stim_chosen]
        self.conf_values[self.stim_chosen] += self.alpha_c_wo * self.conf_PE
        self.values[self.stim_chosen] += self.gamma * self.conf_PE

        return self.values[self.stim_chosen]


class RescorlaConfGenGamma(RescorlaConfGen):
    """Gamma 1 & alpha_c 2 are implemented to capture update differences in phases with & without feedback"""

    def __init__(self, alpha=0.1, beta=1, alpha_c_f=0.1, alpha_c_wo=0.1, gamma=0.1, nbandits=5):

        super().__init__(alpha=alpha, beta=beta, gamma=gamma, nbandits=nbandits)

        self.alpha_c_f = alpha_c_f
        self.alpha_c_wo = alpha_c_wo

    def update(self, outcome, confidence):

        if np.isnan(outcome):
            return self.learn_confidence_value_without_feed(confidence)
        else:
            self.learn_confidence_value_with_feed(confidence)

            return self.learn_value(outcome)

    def learn_confidence_value_with_feed(self, confidence):
        """generic (overall) confidence estimate is used rather than distinct confidence values for each bandit"""

        self.conf_PE = confidence - self.conf_values
        self.conf_values += self.alpha_c_f * self.conf_PE
        self.values[self.stim_chosen] += self.gamma * self.conf_PE

        return self.values[self.stim_chosen]

    def learn_confidence_value_without_feed(self, confidence):
        """generic (overall) confidence estimate is used rather than distinct confidence values for each bandit"""

        self.conf_PE = confidence - self.conf_values
        self.conf_values += self.alpha_c_wo * self.conf_PE
        self.values[self.stim_chosen] += self.gamma * self.conf_PE

        return self.values[self.stim_chosen]
