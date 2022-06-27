import os
from pathlib import Path
from scipy.stats import linregress

import numpy as np
import pandas as pd

from ConfLearning.util.model_to_latex import latex_to_png
from regression import regression

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

data2 = pd.DataFrame(index=range(nsubjects*nblocks*5))
data2['subject'] = np.repeat(range(nsubjects), nblocks*5)
data2['block'] = np.tile(np.repeat(range(nblocks), 5), nsubjects)
data2['bandit'] = np.tile(range(5), nblocks*nsubjects)

for s in range(nsubjects):
    print(f'Subject {s + 1} / {nsubjects}')
    for b in range(nblocks):
        cond = (data.subject == s) & (data.block == b)
        for p in ['b_designated_absvaluediff', 'b_valuebase', 'b_stimulus_pool', 'b_ntrials_pre', 'b_ntrials_noc', 'absvaluediff', 'value_chosen']:
            data2.loc[(data2.subject == s) & (data2.block == b), p] = data[cond & (data.phase == 1) & ~data.equal_value_pair & data.type_choice][p].mean()
        for c in range(5):
            if len(data[cond & (data.phase == 1) & (data.choice == c)]) >= 2:
                data.loc[cond & (data.phase == 1) & (data.choice == c), 'conf_slope'] = linregress(range(len(data[cond & (data.phase == 1) & (data.choice == c)])), np.array(data[cond & (data.phase == 1) & (data.choice == c)].confidence.values, float)).slope
            data.loc[cond & (data.phase == 1) & (data.choice == c), 'choice_valdiff'] = data[cond & (data.phase == 1) & (data.choice == c)].value_chosen - data[cond & (data.phase == 1) & (data.choice == c)].value_unchosen
            data.loc[cond & (data.phase == 1) & (data.choice == c), 'choice_number'] = range(1, len(data[cond & (data.phase == 1) & (data.choice == c)])+1)
            data.loc[cond & (data.phase == 1) & ((data.stimulus_left == c) | (data.stimulus_right == c)), 'occurrence_number'] = range(1, len(data[cond & (data.phase == 1) & ((data.stimulus_left == c) | (data.stimulus_right == c))])+1)
            for p in ['conf_slope', 'choice_valdiff', 'choice_number', 'occurrence_number']:
                data2.loc[(data2.subject == s) & (data2.block == b) & (data2.bandit == c), p] = data[cond & (data.phase == 1) & (data.choice == c) & ~data.equal_value_pair & data.type_choice][p].mean()



map = dict(
    b_designated_absvaluediff='block_difficulty',
    b_valuebase='block_value_level',
    b_stimulus_pool='block_stimulus_type',
    b_ntrials_pre='block_ntrials_phase1',
    b_ntrials_noc='block_ntrials_phase2',
    absvaluediff='trial_difficulty',
    value_chosen='trial_value_chosen',
    trial_phase='trial_number'
)
d = data.copy().rename(columns=map)

ps = ['block_difficulty', 'block_value_level', 'block_stimulus_type', 'block_ntrials_phase1', 'block_ntrials_phase2', 'trial_difficulty', 'choice_number', 'choice_valdiff', 'occurrence_number']
model = regression(
    d[~d.conf_slope.isna() & (d.phase == 1) & d.type_choice & ~d.subject.isin(exclude) & ~d.equal_value_pair],
    patsy_string='conf_slope ~ ' + ' + '.join(ps),
    standardize_vars=True,
    ignore_warnings=True,
    reml=False,
    print_data=False
)

d2 = data2.copy().rename(columns=map)
ps = ['block_difficulty', 'block_value_level', 'block_stimulus_type', 'block_ntrials_phase1', 'block_ntrials_phase2', 'trial_difficulty', 'choice_number', 'choice_valdiff']
model = regression(
    d2[~d2.conf_slope.isna() & ~d2.subject.isin(exclude)],
    patsy_string='conf_slope ~ ' + ' + '.join(ps),
    standardize_vars=True,
    ignore_warnings=True,
    reml=False,
    print_data=False
)




# skip_var_hack = 'subject Var            &  0.382 &    0.120 &        &             &        &         \\\\\nblock Var              &  0.160 &    0.023 &        &             &        &         \\\\\n'
# latex_to_png(model, outpath=os.path.join(os.getcwd(), 'regtables', f'{Path(__file__).stem}.png'),
#              title=None, DV='confidence', skip_var_hack=skip_var_hack)

# df = d[['confidence', 'subject', 'block'] + list(map.values())][~d.confidence.isna() & (d.phase == 1) & d.type_choice & ~d.subject.isin(exclude) & ~d.equal_value_pair]
# df.to_csv('data_phase2_confidence.csv')