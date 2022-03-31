#!/fast/home/users/guggenmm_c/python/bin/python3 -u
import numpy as np
import pandas as pd
import sys
sys.path.extend(['/home/matteo/Dropbox/python/confidence/remeta/'])
sys.path.extend(['/fast/users/guggenmm_c/work/Dropbox/'])
sys.path.extend(['/fast/users/guggenmm_c/work/Dropbox/confidence/remeta/'])
from remeta.gendata import simu_data
from remeta import ReMeta
from scipy.stats import pearsonr
from timeit import default_timer
from concurrent.futures import ProcessPoolExecutor as Pool
import socket
from functools import partial
from itertools import product
# import warnings
# warnings.filterwarnings('error')

seed = 1
np.random.seed(seed)

# Ns
####
nsamples = 10000  # number of trials for each subject
nsubjects = 500

# True parameters
#################

true_params = dict(
    noise_sens=0.4,
    thresh_sens=0,
    bias_sens=0,
    noise_meta=0.2,
    # scaling_meta=1,
    # slope_meta=1,
    slope_prenoise_meta=1,
    readout_term_meta=0
)
nparams = len(true_params)

true_param_ranges = dict(
    noise_sens=[0.2, 0.7, 2],
    thresh_sens=[0, 0.2, 0.4],
    bias_sens=[-0.2, 0, 0.2],
    noise_meta=[0.01, 0.1, 0.2, 0.4, 0.8],
    # scaling_meta=[0.8, 1, 1.25],
    slope_prenoise_meta=[0.8, 1, 1.25],
    readout_term_meta=[-0.2, 0, 0.2]
)

params_sweep = dict(
    noise_sens=np.linspace(0.1, 2, nsubjects),
    thresh_sens=np.linspace(0, 0.5, nsubjects),
    bias_sens=np.linspace(-0.8, 0.8, nsubjects),
    noise_meta=np.linspace(0.1, 0.5, nsubjects),
    readout_term_meta=np.linspace(-0.8, 0.8, nsubjects),
    slope_prenoise_meta=np.linspace(0.5, 1.5, nsubjects),
    # scaling_meta=np.linspace(0.5, 1.5, nsubjects)
)
nsweeps_tot = np.sum([len(v) for v in params_sweep.values()])
nsweep_params = len(params_sweep)

options = dict(
    enable_warping_sens=0,
    enable_noise_sens=1,
    enable_noise_transform_sens=0,
    enable_thresh_sens=1,
    enable_bias_sens=1,
    enable_noise_meta=1,
    enable_noise_transform_meta=0,
    enable_readout_term_meta=1,
    enable_slope_meta=0,
    enable_slope_prenoise_meta=1,
    enable_criteria_meta=0,
    enable_levels_meta=0,
    enable_scaling_meta=0,
    meta_link_function='probability_correct',
    meta_noise_model='truncated_norm',
    # meta_noise_type='noise_readout',
    meta_noise_type='noisy_report',
    gridsearch=False,
    fine_gridsearch=False,
    grid_multiproc=False,
    global_minimization=False,
    skip_meta=False,
    print_configuration=False,
    force_settings=True,
    gradient_free=False,
    binsize_meta=1e-3,
)

print('new')
print(f"meta_noise_type={options['meta_noise_type']}, meta_noise_model={options['meta_noise_model']}, nsamples={nsamples}, nsubjects={nsubjects}")

verbose = False
ignore_warnings = True

def loop(range_sweep, param_sweep, params_start, k):
    params = params_start.copy()
    params[param_sweep] = range_sweep[k]

    m = simu_data(1, nsamples, params.copy(), **{**options, **dict(verbose=False)}, squeeze=True)

    rem = ReMeta(**{**options, 'true_params': params})
    rem.fit(m.stimuli, m.choices, m.confidence, verbose=verbose,
            ignore_warnings=ignore_warnings)
    reconstruction = rem.summary().model
    predicted_params = np.full(nparams, np.nan)
    for l, p in enumerate(true_params.keys()):
        if p in reconstruction.params_sens:
            predicted_params[l] = reconstruction.params_sens[p]
        elif p in reconstruction.params_meta:
            predicted_params[l] = reconstruction.params_meta[p]

    return predicted_params


def main(i):
    t1 = default_timer()
    np.random.seed(i + 1)

    params_start = {k: combos[i][j] for j, k in enumerate(true_params.keys())}

    cm = np.full((nparams, nparams), np.nan)
    for j, (param_sweep, range_sweep) in enumerate(params_sweep.items()):
        t2 = default_timer()
        with Pool(32) as pool:
            predicted_params = list(pool.map(partial(loop, range_sweep, param_sweep, params_start), range(len(range_sweep))))
        # predicted_params = [None] * len(range_sweep)
        # for z in range(len(range_sweep)):
        #     predicted_params[z] = loop(range_sweep, param_sweep, params_start, z)

        cm[j] = [pearsonr(range_sweep, row)[0] for row in np.array(predicted_params).T]
        print(f'\t\tInner loop {j + 1} / {nparams}: {default_timer() - t2:.1f} secs')

    print(f'\tFinished combo {i + 1} / {ncombos}: {default_timer() - t1:.1f} secs')
    return cm

if __name__ == '__main__':

    bulk_id = int(sys.argv[1])
    # bulk_id = 0

    combo_range = range(bulk_id*45, (bulk_id + 1)*45)
    print(f'[bulk_id={bulk_id}] {combo_range}')

    combos = list(product(*[v for v in true_param_ranges.values()]))
    ncombos = len(combos)

    t0 = default_timer()
    for i, combo_id in enumerate(combo_range):
        print(f'\tStarting combo {combo_id + 1} / {ncombos} [{i + 1} / {len(combo_range)}]')
        result = main(combo_id)
        np.save(f"/fast/users/guggenmm_c/work/Dropbox/confidence/metanoise/data/matrix/grid_{options['meta_noise_type']}_{options['meta_noise_model']}_n{nsamples}_{combo_id:03g}.npy", result)

    print(f'\n++++++ Elapsed time: {default_timer() - t0:.1f} secs ++++++\n')
