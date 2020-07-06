import numpy as np
import scipy as stats
from scipy.optimize import minimize


class ParameterFit:

    def __init__(self):
        """Initializes parameters for model fitting."""

        self.subj = None
        self.model = None
        self.run_model = None

        self.nparams = None
        self.data = None

        self.choice_probab = None
        self.negll = None

    def set_model(self, subj, nsubjects, model, run_model, nparams):
        """Set subject and model for parameter estimation."""

        self.subj = subj
        self.model = model
        self.run_model = run_model

        self.nparams = nparams
        self.data = np.full((nsubjects, nparams), np.nan, float)

    def neg_log_likelihood(self, params):
        """ Negative log likelihood for choice probability. """

        self.choice_probab = self.run_model(self.model, params, self.subj)

        self.negll = -np.sum(np.log(np.maximum(self.choice_probab, 1e-8)))

        return self.negll

    def local_minima(self, expect, bounds):
        """ Returns optimized parameters alpha, beta, alpha_c and gamma."""

        self.data[self.subj] = stats.optimize.minimize(self.neg_log_likelihood, x0=expect, bounds=bounds).x

        return self.data

    def model_fit(self, neg_ll):
        """Computes BIC and AIC model fit."""

        aic = 2 * self.nparams + 2 * neg_ll

        bic = self.nparams * np.log(len(self.choice_probab)) + 2 * neg_ll

        return aic, bic
