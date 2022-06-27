import numpy as np
from glob import glob
import os
import pandas as pd
from itertools import product

np.random.seed(0)

root = '/home/matteo/Dropbox/python/confidence/ConfLearning/results/'

reload = False

models = ['Static', 'Deval', 'Choice', 'ConfSpec', 'ConfUnspec', 'Perseveration']
nmodels = len(models)
nsubjects = 250
params = dict(
    Static=['alpha', 'beta'],
    Deval=['alpha', 'beta', 'alpha_n'],
    Choice=['alpha', 'beta', 'lambd'],
    ConfSpec=['alpha', 'beta', 'alpha_c', 'gamma'],
    ConfUnspec=['alpha', 'beta', 'alpha_c', 'gamma'],
    Perseveration=['alpha', 'beta', 'eta']
)

if reload:
    files = sorted(glob(os.path.join(root, 'model_recov', f'model_recov*.pkl.gz')))
    nsubjects = 250
    model_names = ['Static', 'Deval', 'Choice', 'ConfSpec', 'ConfUnspec', 'Perseveration']
    factor = [25, 5, 5, 1, 1, 5]
    dfs = [None] * len(files)
    for k, file in enumerate(files):
        g, f = int(file[file.find('_g')+2]), int(file[file.find('_f')+2])
        d = pd.read_pickle(file)
        ds = [None] * factor[g]
        for i in range(factor[g]):
            ds[i] = d.copy()
            ds[i].subject += i*nsubjects
        dfs[k] = pd.concat(ds).reset_index(drop=True)
        if g == 5:
            dfs[k] = dfs[k][dfs[k].subject < 1042]
            ind = dfs[k].index.values
            dfs[k] = dfs[k].loc[np.setdiff1d(ind, np.random.choice(ind, 50))]
        print(g, f, len(dfs[k]))
    df = pd.concat(dfs).reset_index(drop=True)
    df.to_pickle(os.path.join(root, 'model_recov.pkl.gz'))
else:
    df = pd.read_pickle(os.path.join(root, 'model_recov.pkl.gz'))


nalpha, nbeta = len(df.alpha_id.unique()), len(df.beta_id.unique())
p_fit_gen = np.full((nmodels, nmodels), np.nan)
p_fit_gen_ab = np.full((nalpha, nbeta, nmodels, nmodels), np.nan)
bestfit = [[] for _ in range(nmodels)]
bestfit_ab = [[[[] for _ in range(nmodels)] for j in range(nbeta)] for i in range(nalpha)]
for g in range(nmodels):
    print(f'Genmodel {g + 1} / {nmodels}')
    d = df[(df.gen_model_id == g)]
    nsubjects = d.subject.max()
    pcombos = list(product(*[range(len(d[p].unique())) for p in params[models[g]]]))
    count = np.zeros(nmodels)
    count_ab = np.zeros((nalpha, nbeta, nmodels))
    for j, pcombo in enumerate(pcombos):
        if np.mod(j + 1, 100) == 0:
            print(f'\tCombi {j + 1} / {len(pcombos)}')
        for s in range(nsubjects):
            d_ = d.query(f'subject=={s} & ' + ' & '.join([f'{p}_id=={pcombo[i]}' for i, p in enumerate(params[models[g]])]))
            if len(d_) > 0:
                idxmin = d_.AIC.idxmin()
                best_model = d_.loc[idxmin].fit_model_id
                bestfit[best_model] += [idxmin]
                count[best_model] += 1
                ab = pcombo[:2]
                bestfit_ab[ab[0]][ab[1]][best_model] += [idxmin]
                count_ab[ab[0], ab[1], best_model] += 1
    p_fit_gen[g] = count / np.sum(count)
    p_fit_gen_ab[:, :, g] = count_ab / np.moveaxis(np.tile(count_ab.sum(axis=2), (nmodels, 1, 1)), 0, 2)

np.save(os.path.join(root, 'p_fit_gen.npy'), p_fit_gen)
print(f"Saved to {os.path.join(root, 'p_fit_gen.npy')}")
np.save(os.path.join(root, 'p_fit_gen_ab.npy'), p_fit_gen_ab)
print(f"Saved to {os.path.join(root, 'p_fit_gen_ab.npy')}")

p_gen_fit = np.full((nmodels, nmodels), np.nan)
p_gen_fit_ab = np.full((nalpha, nbeta, nmodels, nmodels), np.nan)
for f in range(nmodels):
    d = df[df.index.isin(bestfit[f])]
    count = np.array([len(d[d.gen_model_id == g]) for g in range(nmodels)])
    p_gen_fit[f] = count / np.sum(count)
    for a in range(nalpha):
        for b in range(nbeta):
            d = df[df.index.isin(bestfit_ab[a][b][f])]
            if len(d) > 0:
                count = np.array([len(d[d.gen_model_id == g]) for g in range(nmodels)])
                p_gen_fit_ab[a, b, f] = count / np.sum(count)
np.save(os.path.join(root, 'p_gen_fit.npy'), p_gen_fit)
print(f"Saved to {os.path.join(root, 'p_gen_fit.npy')}")
np.save(os.path.join(root, 'p_gen_fit_ab.npy'), p_gen_fit_ab)
print(f"Saved to {os.path.join(root, 'p_gen_fit_ab.npy')}")