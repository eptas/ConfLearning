import os
import sys
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sns

# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))
from plot_util import set_fontsize, savefig  # noqa
from ConfLearning.stats.regression import linear_regression

path_data = os.path.join(Path.cwd(), '../data/')

data = pd.read_pickle(os.path.join(path_data, 'data.pkl'))
data = data[~data.subject.isin([25, 30])]

ntrials_phase0 = (9, 12, 15, 18)
ntrials_phase1 = (0, 5, 10, 15)
ntrials_phase2 = (9, 12, 15, 18)
nt_phase1_max = np.max(ntrials_phase1)
nt_phase2_max = np.max(ntrials_phase2)
nt_phase0phase1 = 27

colors = sns.color_palette()


m, se = np.full(4, np.nan), np.full(4, np.nan)
rating_diff_m, rating_diff_se = np.full(4, np.nan), np.full(4, np.nan)
confslope_m, confslope_se = np.full(4, np.nan), np.full(4, np.nan)
ps = ['block', 'b_designated_absvaluediff', 'b_stimulus_pool', 'b_ntrials_pre', 'b_ntrials_noc', 'value_chosen', 'b_valuebase']
model = linear_regression(
    data[~data.ratingdiff21.isna()],
    patsy_string='ratingdiff21 ~ ' + ' + '.join(ps),
    # patsy_string='ratingdiff21 ~ ' + ' + '.join(ps) + ' + value_chosen*b_valuebase',
    standardize_vars=True,
    ignore_warnings=True,
    model_blocks=False,
    reml=False,
    print_data=False,
    silent=True
)
m[0] = model.params['value_chosen']
se[0] = model.bse['value_chosen']
rating_diff_m[0] = data.ratingdiff21.mean()
rating_diff_se[0] = data.groupby('subject').ratingdiff21.mean().sem()
confslope_m[0] = data.confslope.mean()
confslope_se[0] = data.groupby('subject').confslope.mean().sem()

for i, nt in enumerate(ntrials_phase1[1:]):
    ps = ['block', 'b_designated_absvaluediff', 'b_stimulus_pool', 'b_ntrials_pre', 'value_chosen', 'b_valuebase']
    model = linear_regression(
        data[~data.ratingdiff21.isna() & (data.b_ntrials_noc == nt)],
        patsy_string='ratingdiff21 ~ ' + ' + '.join(ps),
        # patsy_string='ratingdiff21 ~ ' + ' + '.join(ps) + ' + value_chosen*b_valuebase',
        standardize_vars=True,
        ignore_warnings=True,
        model_blocks=False,
        reml=False,
        print_data=False,
        silent=True
    )
    m[i+1] = model.params['value_chosen']
    se[i+1] = model.bse['value_chosen']
    rating_diff_m[i+1] = data[data.b_ntrials_noc == nt].ratingdiff21.mean()
    rating_diff_se[i+1] = data[data.b_ntrials_noc == nt].groupby('subject').ratingdiff21.mean().sem()
    confslope_m[i+1] = data[data.b_ntrials_noc == nt].confslope.mean()
    confslope_se[i+1] = data[data.b_ntrials_noc == nt].groupby('subject').confslope.mean().sem()

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
savefig('../figures/behav/ratingdiff_value.png')