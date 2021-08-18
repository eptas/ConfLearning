import os
from pathlib import Path

import numpy as np
import pandas as pd

from ConfLearning.util.model_to_latex import latex_to_png
from regression import regression
import arviz as az
import bambi as bmb

path_data = os.path.join(Path.cwd(), '../data/')

data = pd.read_pickle(os.path.join(path_data, 'data.pkl'))

nsubjects = 66
nblocks = 11
npairs = 10
ntrials_phase0 = (9, 12, 15, 18)
ntrials_phase1 = (0, 5, 10, 15)
ntrials_phase2 = (9, 12, 15, 18)

# We're including subjects with at least 55% performance
include = np.where(np.array(100*data.groupby('subject').correct.mean().values, int) > 55)[0]
exclude = np.setdiff1d(range(nsubjects), include)
print(f"Subjects with performance < 0.55 (N={len(exclude)}, remain={nsubjects - len(exclude)}): [{', '.join([str(v) for v in exclude])}]")


map = dict(
    b_designated_absvaluediff='block_difficulty',
    b_valuebase='block_value_level',
    b_stimulus_pool='block_stimulus_type',
    absvaluediff='trial_difficulty',
    value_chosen='trial_value_chosen',
    trial_prepost='trial_number'
)
d = data.copy().rename(columns=map).astype(dict(correct=float))
ps = ['block_difficulty', 'block_value_level', 'block_stimulus_type', 'trial_difficulty', 'trial_value_chosen', 'trial_number']
d = d.astype({v: float for v in ps})
patsy_string = 'correct ~ ' + ' + '.join(ps)
for p in ps:
    d[p] = (d[p] - d[p].mean()) / d[p].std()
fitted = bmb.Model(patsy_string, d[~d.correct.isna() & ~d.trial_number.isna()], family="bernoulli").fit()
print(az.summary(fitted, kind='stats'))
# model = bmb.Model(patsy_string, d[~d.correct.isna()])
# fitted = model.fit()

#
# latex_to_png(model, outpath=os.path.join(os.getcwd(), 'regtables', f'{Path(__file__).stem}.png'),
#              title=None, DV='correct')