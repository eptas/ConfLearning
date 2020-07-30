import numpy as np
import scipy as stats
from scipy.optimize import minimize, fmin, basinhopping, fmin_slsqp
from scipy.optimize import Bounds, LinearConstraint, minimize, SR1


class RandomDisplacementBounds(object):
    """random displacement with bounds"""

    def __init__(self, xmin, xmax, stepsize=0.5):

        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds"""

        while True:
            xnew = x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
            if np.all(xnew < self.xmax) and np.all(xnew > self.xmin):
                break

        return xnew


class ParameterFit:

    def __init__(self):

        self.subj = None
        self.model = None
        self.run_model = None

        self.nparams = None
        self.data = None

        self.choice_probab = None
        self.negll = None

    def set_model(self, subj, nsubjects, model, run_model, nparams):
        """set subject and model for parameter estimation"""

        self.subj = subj
        self.model = model
        self.run_model = run_model

        self.nparams = nparams
        self.data = np.full((nsubjects, nparams), np.nan, float)

    def local_minima(self, expect, bounds):
        """returns optimized parameters as well as the negative log-likelihood using optimize function"""

        # unbounded optimization: quite fast and finds good minima:
        self.data[self.subj], self.negll = fmin(self.run_model, expect, args=(self.model, self.subj), full_output=True, disp=False)[:2]

        # scipy's minimize: seems to get stuck in local minima - maybe some parameters would have to be tweaked to make this work:
        # result = stats.optimize.minimize(self.run_model, args=(self.model, self.subj), x0=expect, bounds=bounds, method='L-BFGS-B', options=dict(disp=False))
        # self.data[self.subj], self.negll = result.x, result.fun

        # basinhopping: slow, but finds good minima and respects bounds:
        # result = basinhopping(self.run_model, expect, minimizer_kwargs=dict(method="L-BFGS-B", bounds=bounds, args=(self.model, self.subj)))
        # specifying the take_step variable as below, enforces the bounds more strictly, see https://stackoverflow.com/a/21967888/2320035
        # result = basinhopping(self.run_model, expect, minimizer_kwargs=dict(method="L-BFGS-B", bounds=bounds, args=(self.model, self.subj)), take_step=RandomDisplacementBounds(bounds[:, 0], bounds[:, 1]))
        # self.data[self.subj], self.negll = result.x, result.fun

        return self.data, self.negll

    def model_fit(self, neg_ll, nsamples):
        """computes BIC and AIC model fit"""

        aic = 2 * self.nparams + 2 * neg_ll
        bic = self.nparams * np.log(nsamples) + 2 * neg_ll

        return aic, bic
