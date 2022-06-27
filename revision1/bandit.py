import numpy as np
from scipy.stats import truncnorm


class BanditMoney:
    def __init__(self, outcome_sigma=10, outcome_max=50, maxdev=10, nstim=5):

        self.outcome_sigma = outcome_sigma
        self.outcome_max = outcome_max
        self.maxdev = maxdev
        self.nstim = nstim

        self.outcome_means = None
        self.outcome_diff = None
        self.outcome_history = None
        self.outcome_schedule = None
        self.equal_stimuli = None

    def sample(self, choice, ignore_history_constraints=False):
        valid_range = list(range(self.outcome_means[choice] - self.maxdev, self.outcome_means[choice] + self.maxdev + 1))

        # make sure that equal-value stimuli have approximately equal value
        if choice in self.equal_stimuli and (ignore_history_constraints or (len(self.outcome_history[self.equal_stimuli[0]]) and len(self.outcome_history[self.equal_stimuli[1]]) and (len(self.outcome_history[self.equal_stimuli[0]]) != len(self.outcome_history[self.equal_stimuli[1]])))):
            other_stim = self.equal_stimuli[self.equal_stimuli.index(choice) == 0]
            new_deviations = [np.abs(np.mean(self.outcome_history[other_stim] if self.outcome_history[other_stim] else self.outcome_means[other_stim]) - np.hstack((self.outcome_history[choice], r)).mean()) for r in valid_range]
            outcome = valid_range[np.argmin(new_deviations)]
        # make sure that random sampling doesn't produce values too far from the mean
        elif len(self.outcome_history[choice]) and (np.abs(np.mean(self.outcome_history[choice]) - self.outcome_means[choice]) > (self.outcome_sigma / np.sqrt(len(self.outcome_history[choice])))):
            new_deviations = [np.abs(self.outcome_means[choice] - np.hstack((self.outcome_history[choice], r)).mean()) for r in valid_range]
            outcome = valid_range[np.argmin(new_deviations)]
        else:
            outcome = truncnorm.rvs(-self.maxdev / self.outcome_sigma, self.maxdev / self.outcome_sigma, loc=self.outcome_means[choice], scale=self.outcome_sigma)
        outcome = max(0, min(self.outcome_max, int(np.round(outcome))))

        self.outcome_history[choice] = self.outcome_history[choice] + [outcome]

        return outcome

    def reset_outcome_history(self):
        self.outcome_history = [[]] * self.nstim

    def reset(self):
        self.reset_outcome_history()
        self.outcome_means = None
        self.outcome_diff = None
        self.outcome_history = None
        self.outcome_schedule = None
        self.equal_stimuli = None

    def set_outcome_schedule(self, outcome_schedule, outcome_base, outcome_diff):
        self.outcome_schedule = outcome_schedule
        self.outcome_base = outcome_base
        self.outcome_diff = outcome_diff
        idx = list(np.unique(self.outcome_schedule, return_counts=True)[1]).index(2)
        self.equal_stimuli = [idx, idx + 1]
        self.outcome_means = self.outcome_base + self.outcome_schedule * self.outcome_diff

if __name__ == '__main__':
    bandit = BanditMoney()