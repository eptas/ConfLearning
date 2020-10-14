import numpy as np
import scipy as stats
from scipy.optimize import minimize, fmin, basinhopping, fmin_slsqp
from scipy.optimize import Bounds, LinearConstraint, minimize, SR1
from itertools import product
from multiprocessing import Pool, cpu_count
from functools import partial


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

    def set_model(self, subj, nsubjects, model, run_model, nparams):
        """set subject and model for parameter estimation"""

        self.subj = subj
        self.model = model
        self.run_model = run_model

        self.nparams = nparams
        self.data = np.full((nsubjects, nparams), np.nan, float)

    def local_minima(self, expect, bounds, grid_range, grid_multiproc=True):
        """returns optimized parameters as well as the negative log-likelihood using optimize function"""

        args = (self.model, self.subj)

        combis = list(product(*grid_range))
        if grid_multiproc:
            with Pool(cpu_count() - 1 or 1) as pool:
                nll_grid = pool.map(partial(loop, self.run_model, args), combis)
        else:
            nll_grid = [None] * len(combis)
            for i, param in enumerate(combis):
                nll_grid[i] = self.run_model(param, *args)
        expect = combis[np.argmin(nll_grid)]

        # unbounded optimization: quite fast and finds good minima:
        # params_fmin, negll_fmin = fmin(self.run_model, expect, args=args, full_output=True, disp=False)[:2]
        fit_lbfgs = stats.optimize.minimize(self.run_model, args=args, x0=expect, bounds=bounds, method='L-BFGS-B', options=dict(disp=False))
        params_lbfgs, negll_lbfgs = fit_lbfgs.x, fit_lbfgs.fun
        fit_powell = stats.optimize.minimize(self.run_model, args=args, x0=expect, bounds=bounds, method='Powell', options=dict(disp=False))
        params_powell, negll_powell = fit_powell.x, fit_powell.fun
        print(f'L-BFGS-B: {negll_lbfgs:.2f}, Powell: {negll_powell:.2f}')
        print(f'L-BFGS-B: {params_lbfgs}, Powell: {params_powell}')

        params = (params_lbfgs, params_powell)
        negll = (negll_lbfgs, negll_powell)
        self.negll = np.min(negll)
        self.data[self.subj] = params[np.argmin(negll)]

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
