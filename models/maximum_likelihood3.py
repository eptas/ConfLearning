import numpy as np
import scipy as stats
from scipy.optimize import minimize, fmin, basinhopping, fmin_slsqp
from scipy.optimize import Bounds, LinearConstraint, minimize, SR1
from itertools import product
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
import dill
from multiprocessing_on_dill.pool import Pool as DillPool
from pathos.multiprocessing import ProcessingPool as PathosPool
from functools import partial

# dill.settings['recurse'] = True

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


def loop(fun, args, params):
    return fun(params, *args)


class ParameterFit:

    def __init__(self):

        self.subj = None
        self.model = None
        self.run_model = None

        self.nparams = None
        self.data = None

        self.choice_probab = None
        self.negll = None

    def set_model(self, subj, nsubjects, model, run_model, nparams, sim_model=None):
        """set subject and model for parameter estimation"""

        self.subj = subj
        self.model = model
        self.run_model = run_model

        self.nparams = nparams
        self.data = np.full((nsubjects, nparams), np.nan, float)
        self.sim_model = sim_model

    def local_minima(self, expect, bounds, grid_range, grid_multiproc=True, verbose=True, args=None):
        """returns optimized parameters as well as the negative log-likelihood using optimize function"""

        if args is None:
            args = (self.model, self.subj, self.sim_model)
        else:
            args = tuple([self.model, self.subj, self.sim_model] + args)

        combis = list(product(*grid_range))
        if grid_multiproc:
            with PathosPool(cpu_count() - 1 or 1) as pool:
                nll_grid = pool.map(partial(loop, self.run_model, args), combis)
        else:
            nll_grid = [None] * len(combis)
            for i, param in enumerate(combis):
                nll_grid[i] = self.run_model(param, *args)
        expect = combis[np.argmin(nll_grid)]

        # unbounded optimization: quite fast and finds good minima:
        # params_fmin, negll_fmin = fmin(self.run_model, expect, args=args, full_output=True, disp=False)[:2]
        fit_lbfgs = stats.optimize.minimize(self.run_model, x0=np.array(expect), args=args, bounds=bounds, method='L-BFGS-B', options=dict(disp=False))
        self.data[self.subj], self.negll = fit_lbfgs.x, fit_lbfgs.fun

        return self.data, self.negll

    def model_fit(self, neg_ll, nsamples):
        """computes BIC and AIC model fit"""

        aic = 2 * self.nparams + 2 * neg_ll
        bic = self.nparams * np.log(nsamples) + 2 * neg_ll

        return aic, bic
