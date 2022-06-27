#!/fast/home/users/ptasczle_c/python/bin/python3 -u
import os
import sys
from timeit import default_timer

import numpy as np
import pandas as pd
from scipy.stats import linregress

HOME = os.path.expanduser("~")

if os.path.exists(os.path.join(HOME, 'work/Dropbox/confidence/ConfLearning/results/')):
    data_root = os.path.join(HOME, 'work/Dropbox/confidence/ConfLearning/results/')
    root = os.path.join(HOME, 'work/Dropbox/confidence/ConfLearning/data/')
else:
    data_root = '/home/matteo/outsource_Dropbox/'
    root = '/home/matteo/Dropbox/python/confidence/ConfLearning/data/'

model_names = ['ChoiceMono', 'ConfBase', 'ConfBaseGen', 'Perseveration']
# model_names = ['ConfBaseGen']

nmodels = len(model_names)
nalphac = 3
ngamma = 8

nsubjects = 100
nblocks = 11
nbulk = 100

if __name__ == '__main__':

    if len(sys.argv) > 1:
        bulk_id = int(sys.argv[1])
    else:
        bulk_id = 0

    res = pd.DataFrame(index=range(nmodels * nalphac * ngamma * nsubjects))
    res['model_id'] = np.repeat(range(nmodels), nalphac * ngamma * nsubjects)
    res['alpha_c_id'] = np.tile(np.repeat(range(nalphac), ngamma * nsubjects), nmodels)
    res['gamma_id'] = np.tile(np.repeat(range(ngamma), nsubjects), nmodels * nalphac)
    res['subject'] = bulk_id*nsubjects + np.tile(range(nsubjects), nalphac * ngamma * nmodels)

    print(os.path.join(data_root, f'behav_modulation_{bulk_id}.pkl.gz'))
    data = pd.read_pickle(os.path.join(data_root, f'behav_modulation_{bulk_id}.pkl.gz'))
    data = data[data.type_choice & ~data.equal_value_pair & (data.phase == 1)].astype(dict(correct=int))
    for m, mod in enumerate(model_names):
        dm = data[data.model_id == m]
        for a in range(nalphac):
            da = dm[dm.alpha_c_id == a]
            for g in range(ngamma):

                print(f'BulkID {bulk_id + 1} / {nbulk} Model {m + 1} / {nmodels} alpha_c {a + 1} / {nalphac} gamma {g + 1} / {ngamma}')

                d = da[da.gamma_id == g]
                cond = (res.model_id == m) & (res.gamma_id == g) & (res.alpha_c_id == a)
                res.loc[cond, 'model'] = d.model.values[0]
                res.loc[cond, 'alpha_c'] = d.alpha_c.values[0]
                res.loc[cond, 'gamma'] = d.gamma.values[0]

                if ~np.isnan(d.gamma.values[0]):
                    t0 = default_timer()
                    for s in range(nsubjects):

                        scond = cond & (res.subject == bulk_id*nsubjects + s)
                        ds = d[d.subject == s]

                        index = ds.groupby('trial_phase').correct.mean().index.values
                        performances = ds.groupby('trial_phase').correct.mean().values
                        res.loc[scond, 'perf_slope'] = linregress(index, performances)[0]
                        confidences = ds.groupby('trial_phase').confidence.mean().values
                        res.loc[scond, 'conf_slope'] = linregress(index, confidences)[0]

                        confslopes = np.full(4, np.nan)
                        confslopes_full = np.full(4, np.nan)
                        for ch in range(0, 4):
                            conf_ = ds[ds.value_order == ch].groupby('trial_phase').confidence.mean()
                            # index = conf_.index.values
                            index = range(len(conf_))
                            confidences = conf_.values
                            if len(index) >= 2:
                                confslopes[ch] = linregress(index, confidences)[0]
                                res.loc[scond, f'conf_slope{ch}'] = confslopes[ch]
                            confslopes_full[ch] = linregress(range(15), ds.groupby('trial_phase')[f'confidence{ch}'].mean())[0]
                            res.loc[scond, f'conf_slope{ch}_full'] = confslopes_full[ch]
                        if np.sum(~np.isnan(confslopes)) >= 2:
                            res.loc[scond, 'conf_value_slope'] = linregress(np.where(~np.isnan(confslopes))[0], confslopes[~np.isnan(confslopes)])[0]
                        res.loc[scond, 'conf_value_slope_full'] = linregress(range(4), confslopes_full)[0]

                        count, consistent = 0, 0
                        count2, consistent2 = 0, 0
                        count3, consistent3 = 0, 0
                        for b in range(nblocks):
                            for p in ds.pair.unique():
                                if len(ds[ds.pair == p]) > 1:
                                    trials = ds[(ds.block == b) & (ds.pair == p)].trial_phase.values
                                    for i, t in enumerate(trials[1:]):
                                        count += 1
                                        consistency = ds[(ds.block == b) & (ds.trial_phase == t)].choice.values[0] == ds[(ds.block == b) & (ds.trial_phase == trials[i])].choice.values[0]
                                        consistent += consistency
                                        if i == 0:
                                            count2 += 1
                                            consistent2 += consistency
                                        elif i == 1:
                                            count3 += 1
                                            consistent3 += consistency
                        res.loc[scond, 'consistency'] = consistent / count
                        res.loc[scond, 'consistency12'] = consistent2 / count2
                        res.loc[scond, 'consistency23'] = consistent3 / count3
                        res.loc[scond, 'consistency_diff'] = consistent3 / count3 - consistent2 / count2

                    print(f'\t{default_timer() - t0:.2f} secs')

    res.to_pickle(os.path.join(root, '../data/sim', f'sim_{bulk_id:03g}.pkl.gz'))
    print(f"Saved to {os.path.join(root, '../data/sim', f'sim_{bulk_id:03g}.pkl.gz')}")