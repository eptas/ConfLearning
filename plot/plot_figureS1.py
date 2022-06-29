import os
import sys
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))
from ConfLearning.plot.util.plot_util import set_fontsize, savefig  # noqa
from ConfLearning.stats.regression import regression

path = Path(__file__).parent
data = pd.read_pickle(os.path.join(path, '../data/', 'data.pkl'))

data = data[~data.subject.isin([25, 30])]

ntrials_phase0 = (9, 12, 15, 18)
ntrials_phase1 = (0, 5, 10, 15)
ntrials_phase2 = (9, 12, 15, 18)
nt_phase1_max = np.max(ntrials_phase1)
nt_phase2_max = np.max(ntrials_phase2)
nt_phase0phase1 = 27

colors = sns.color_palette()

map = dict(
    b_designated_absvaluediff='block_difficulty',
    b_valuebase='block_value_level',
    b_ntrials_pre='block_ntrials_phase1',
    b_ntrials_noc='block_ntrials_phase2',
    b_stimulus_pool='block_stimulus_type',
    value_chosen='value',
    ratingdiff21='rating_change',
)
d = data.copy().rename(columns=map)

m, se = np.full(4, np.nan), np.full(4, np.nan)
rating_diff_m, rating_diff_se = np.full(4, np.nan), np.full(4, np.nan)
confslope_m, confslope_se = np.full(4, np.nan), np.full(4, np.nan)
ps = ['block_difficulty', 'block_value_level', 'block_stimulus_type', 'block_stimulus_type', 'block_ntrials_phase1', 'block_ntrials_phase2', 'value']
model = regression(
    d[~d.rating_change.isna()],
    patsy_string='rating_change ~ ' + ' + '.join(ps),
    # patsy_string='ratingdiff21 ~ ' + ' + '.join(ps) + ' + value_chosen*b_valuebase',
    standardize_vars=True,
    ignore_warnings=True,
    model_blocks=False,
    reml=False,
    print_data=False,
    silent=True
)
m[0] = model.params['value']
se[0] = model.bse['value']
rating_diff_m[0] = d.rating_change.mean()
rating_diff_se[0] = d.groupby('subject').rating_change.mean().sem()
confslope_m[0] = d.confslope.mean()
confslope_se[0] = d.groupby('subject').confslope.mean().sem()

for i, nt in enumerate(ntrials_phase1[1:]):
    ps = ['block_difficulty', 'block_value_level', 'block_stimulus_type', 'block_stimulus_type', 'block_ntrials_phase1', 'value']
    model = regression(
        d[~d.rating_change.isna() & (d.block_ntrials_phase2 == nt)],
        patsy_string='rating_change ~ ' + ' + '.join(ps),
        # patsy_string='ratingdiff21 ~ ' + ' + '.join(ps) + ' + value_chosen*b_valuebase',
        standardize_vars=True,
        ignore_warnings=True,
        model_blocks=False,
        reml=False,
        print_data=False,
        silent=True
    )
    m[i+1] = model.params['value']
    se[i+1] = model.bse['value']
    rating_diff_m[i+1] = d[d.block_ntrials_phase2 == nt].rating_change.mean()
    rating_diff_se[i+1] = d[d.block_ntrials_phase2 == nt].groupby('subject').rating_change.mean().sem()
    confslope_m[i+1] = d[d.block_ntrials_phase2 == nt].confslope.mean()
    confslope_se[i+1] = d[d.block_ntrials_phase2 == nt].groupby('subject').confslope.mean().sem()

# m, se = rating_diff_m, rating_diff_se
# m, se = confslope_m, confslope_se

plt.figure(figsize=(4, 3))
plt.plot([-0.5, 3.5], [0, 0], 'k-', lw=0.5)
plt.bar(0, m[0], yerr=se[0], facecolor=colors[0])
plt.bar(range(1, 4), m[1:], yerr=se[1:], facecolor=(0.4, 0.4, 0.4))

plt.ylabel('Regression coefficient')
plt.xticks(range(4), ['All'] + [*ntrials_phase1[1:]])
plt.xlabel('Number of trials in phase 2')
plt.title('Effect of value on rating change', fontsize=11)
plt.xlim([-0.5, 3.5])
plt.tight_layout()
set_fontsize(label=11, tick=9)
# savefig('../figures/ratingdiff_lengthphase2.png')
savefig(f'../figures/FigureS1.tif', format='tif', dpi=600, pil_kwargs={"compression": "tiff_lzw"})