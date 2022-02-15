import os
from pathlib import Path

import numpy as np
import pandas as pd

from ConfLearning.util.model_to_latex import latex_to_png
from regression import regression
from itertools import combinations

path_data = os.path.join(Path.cwd(), '../data/')

data = pd.read_pickle(os.path.join(path_data, 'data.pkl'))

nsubjects = 66
nblocks = 11

# We're including subjects with at least 55% performance
include = np.where(np.array(100*data.groupby('subject').correct.mean().values, int) > 55)[0]
exclude = np.setdiff1d(range(nsubjects), include)
print(f"Subjects with performance < 0.55 (N={len(exclude)}, remain={nsubjects - len(exclude)}): [{', '.join([str(v) for v in exclude])}]")

reload = True
if reload:
    path_data = os.path.join(Path.cwd(), '../data/')
    data = pd.read_pickle(os.path.join(path_data, 'data.pkl'))
    stim_combos = list(combinations(range(5), 2))
    ncombos = len(stim_combos)
    df = pd.DataFrame(index=range(nsubjects * nblocks * ncombos * 2))
    df['subject'] = np.repeat(range(nsubjects), nblocks * ncombos * 2)
    df['block'] = np.tile(np.repeat(range(nblocks), ncombos * 2), nsubjects)
    df['pair'] = np.tile(np.repeat(range(ncombos), 2), nsubjects * nblocks)
    df['time'] = np.tile(range(2), nsubjects * nblocks * ncombos)
    for s in range(nsubjects):
        print(f'Subject {s + 1} / {nsubjects}')
        for b in range(nblocks):
            d = data[(data.subject == s) & (data.block == b) & data.b_has_rating1 & data.b_has_rating2]
            if len(d):
                for col in ('b_designated_absvaluediff', 'b_valuebase', 'b_ntrials_pre', 'b_ntrials_noc', 'b_stimulus_pool'):
                    df.loc[(df.subject == s) & (df.block == b), col] = d[col].values[0]
                for i, c in enumerate(stim_combos):
                    absratingsdiff1 = np.abs(d[(d.stimulus_left == c[0]) & d.type_rating1].rating.values[0] - d[(d.stimulus_left == c[1]) & d.type_rating1].rating.values[0])
                    absratingsdiff2 = np.abs(d[(d.stimulus_left == c[0]) & d.type_rating2].rating.values[0] - d[(d.stimulus_left == c[1]) & d.type_rating2].rating.values[0])
                    df.loc[(df.subject == s) & (df.block == b) & (df.pair == i) & (df.time == 0), 'absratingdiff'] = absratingsdiff1
                    df.loc[(df.subject == s) & (df.block == b) & (df.pair == i) & (df.time == 1), 'absratingdiff'] = absratingsdiff2
                    df.loc[(df.subject == s) & (df.block == b) & (df.pair == i), 'absvaluediff'] = np.abs(d[(d.stimulus_left == c[0]) & d.type_rating1].value_chosen.values[0] - d[(d.stimulus_left == c[1]) & d.type_rating1].value_chosen.values[0])
                    df.loc[(df.subject == s) & (df.block == b) & (df.pair == i), 'absvaluesum'] = d[(d.stimulus_left == c[0]) & d.type_rating1].value_chosen.values[0] + d[(d.stimulus_left == c[1]) & d.type_rating1].value_chosen.values[0]
    df.to_pickle('absratingdiff.pkl')
else:
    df = pd.read_pickle('absratingdiff.pkl')


map = dict(
    b_designated_absvaluediff='block_difficulty',
    b_valuebase='block_value_level',
    b_ntrials_pre='block_ntrials_phase1',
    b_ntrials_noc='block_ntrials_phase2',
    b_stimulus_pool='block_stimulus_type',
    ratingdiff21='rating_change',
    pair='CS_pair'
)
d = df.copy().rename(columns=map)


ps = ['block_difficulty', 'block_value_level', 'block_stimulus_type', 'block_ntrials_phase1', 'block_ntrials_phase2', 'absvaluediff', 'absvaluesum', 'C(CS_pair)', 'time']
model = regression(
    d[~d.absratingdiff.isna() & (d.block_ntrials_phase2 > 0)],
    patsy_string='absratingdiff ~ ' + ' + '.join(ps),
    standardize_vars=True,
    ignore_warnings=True,
    model_blocks=True,
    reml=False,
    print_data=False
)
# skip_var_hack = 'subject Var            &  0.023 &    0.035 &        &             &        &         \\\\\nblock Var              &  0.075 &    0.043 &        &             &        &         \\\\\n'
# latex_to_png(model, outpath=os.path.join(os.getcwd(), 'regtables', f'{Path(__file__).stem}.png'),
#              title=None, DV='rating\_change', skip_var_hack=skip_var_hack)