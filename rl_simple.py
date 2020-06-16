import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class RLModel:

    def __init__(self, alpha=0.1, beta=3, nbandits=5):
        """This function defines / initializes (free) parameters for a two-armed bandit task."""

        self.alpha = alpha
        self.beta = beta
        self.nbandits = nbandits

        self.stims = None
        self.stim_chosen = None

        self.trial = None

        self.choice_probab = None
        self.PE = None

        self.confidence = None

        self.values = np.full(self.nbandits, 0, float)
        self.value_history = [[] for _ in range(self.nbandits)]

    def set_stimuli(self, stims):
        """This function sets current stimuli of experimental data."""

        self.stims = stims

    def get_current_trial(self, trial):

        self.trial = trial + 1

    def get_choice_probab(self):
        """Function outputs choice probability for left stimulus."""

        value1, value2 = self.values[self.stims[0]], self.values[self.stims[1]]

        self.choice_probab = 1 / (1 + np.exp(-self.beta * (value2 - value1)))

        return self.choice_probab

    def choice(self):
        """Function outputs actual choice for left stimulus, if below random."""

        self.stim_chosen = self.stims[int(np.random.rand() < self.choice_probab)]

        return self.stim_chosen

    def update(self, outcome, confidence):

        return self.learn_value(outcome)

    def learn_value(self, outcome):
        """This function updates reward values according to Rescorla Wagner learning rule."""

        if ~np.isnan(outcome):
            self.PE = outcome - self.values[self.stim_chosen]

            self.values[self.stim_chosen] += self.alpha * self.PE

        return self.values[self.stim_chosen]

    def learn_history(self):
        """This function stores learning history of all bandit values."""

        self.value_history[self.stim_chosen] = self.value_history[self.stim_chosen] + [self.values[self.stim_chosen]]

        return self.value_history

    def get_confidence(self):

        self.confidence = (np.abs(self.choice_probab - (self.choice_probab < 0.5)) - 0.5) * 2

        return self.confidence


class RLModelWithoutFeedback(RLModel):

    def __init__(self, alpha=0.1, beta=3, alpha_n=0.1, nbandits=5):

        super().__init__(alpha=alpha, beta=beta, nbandits=nbandits)

        self.alpha_n = alpha_n

    def update(self, outcome, confidence):

        if np.isnan(outcome):
            return self.learn_without_outcome()

        else:
            return self.learn_value(outcome)

    def learn_without_outcome(self):

        self.PE = 0 - self.values[self.stim_chosen]

        self.values[self.stim_chosen] += self.alpha_n * self.PE

        return self.values[self.stim_chosen]


class ConfidencePE(RLModel):

    def __init__(self, alpha=0.1, beta=3, alpha_c=0.1, gamma=0.1, nbandits=5):

        super().__init__(alpha=alpha, beta=beta, nbandits=nbandits)

        self.alpha_c = alpha_c
        self.gamma = gamma

        self.conf_values = np.full(self.nbandits, 0, float)
        self.conf_PE = None

    def update(self, outcome, confidence):

        self.conf_PE = confidence - self.conf_values[self.stim_chosen]
        self.conf_values[self.stim_chosen] += self.gamma * self.conf_PE

        if np.isnan(outcome):
            return self.learn_confidence_value(confidence)
        else:
            return self.learn_value(outcome)

    def learn_confidence_value(self, confidence):
        """This function updates participants' expected values according to confidence predition error."""

        self.values[self.stim_chosen] += self.alpha_c * self.conf_PE

        return self.values[self.stim_chosen]


class ConfidencePEgeneric(ConfidencePE):

    def __init__(self, alpha=0.1, beta=3, alpha_c=0.1, gamma=0.1, nbandits=5):

        super().__init__(alpha=alpha, beta=beta, alpha_c=alpha_c, gamma=gamma, nbandits=nbandits)

        self.conf_values = 0

    def learn_confidence_value(self, confidence):
        """This function updates participants' expected values according to confidence predition error."""

        self.conf_PE = confidence - self.conf_values

        self.conf_values += self.gamma * self.conf_PE

        self.values[self.stim_chosen] += self.alpha_c * self.conf_PE

        return self.values[self.stim_chosen]


class ConfidenceIdealObserver(ConfidencePE):

    def __init__(self, alpha=0.1, beta=3, alpha_c=0.1, gamma=0.1, nbandits=5):

        super().__init__(alpha=alpha, beta=beta, alpha_c=alpha_c, gamma=gamma, nbandits=nbandits)

    def update(self, outcome, confidence):

        if np.isnan(outcome):
            return self.learn_confidence_value(confidence)

        else:
            self.track_confidence_value(confidence)

            return self.learn_value(outcome)

    def track_confidence_value(self, confidence):

        self.conf_PE = confidence - self.conf_values[self.stim_chosen]

        self.conf_values[self.stim_chosen] += self.gamma * self.conf_PE


class BayesModel(RLModel):

    def __init__(self, alpha=0.1, beta=3, phi=0.1, gamma=3, nsamples=None):

        super().__init__(alpha=alpha, beta=beta)

        self.phi = phi
        self.gamma = gamma

        self.nsamples = nsamples
        self.values_sigma = np.full(self.nbandits, 1, float)

    def get_choice_probab(self):

        delta_values = self.values[self.stims[0]] - self.values[self.stims[1]]
        delta_values_sigma = np.sqrt(self.values_sigma[self.stims[0]]) - np.sqrt(self.values_sigma[self.stims[1]])

        self.choice_probab = 1 / (1 + np.exp(-self.beta * (delta_values + self.phi * delta_values_sigma)))

        return self.choice_probab

    def update(self, outcome, confidence):

        self.PE = outcome - self.values[self.stim_chosen]
        self.values_sigma[self.stim_chosen] += self.alpha * (self.PE ** 2 - self.values_sigma[self.stim_chosen])
        self.values[self.stim_chosen] += self.alpha * self.PE

        return self.values[self.stim_chosen]

    def get_confidence(self):

        pooled_sd = np.sqrt(np.sum(self.values_sigma))
        smd = (self.values[int(self.stim_chosen == 1)] - self.values[int(self.stim_chosen == 0)]) / pooled_sd

        self.confidence = (1 / (1 + np.exp(-self.gamma * smd)) - 0.5) * 2

        return self.confidence


class BayesIdealObserver(BayesModel):

    def __init__(self, alpha=0.1, beta=3, phi=0.1, gamma=3):

        super().__init__(alpha=alpha, beta=beta, phi=phi, gamma=gamma)

    def update(self, outcome, confidence):

        self.PE = outcome - self.values[self.stim_chosen]
        self.values_sigma[self.stim_chosen] = (1 - 1 / self.trial) * self.values_sigma[self.stim_chosen] + self.PE ** 2 / (1 + self.trial)
        # self.values_sigma[self.stim_chosen] = self.values_sigma[self.stim_chosen] + (self.PE ** 2/( 1 + self.trial) - self.values_sigma[self.stim_chosen] / self.trial)  # equivalent
        self.values[self.stim_chosen] = (self.trial * self.values[self.stim_chosen] + outcome) / (self.trial + 1)

        return self.values[self.stim_chosen]
