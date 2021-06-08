import os
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, wilcoxon, linregress
from itertools import combinations
from regression import linear_regression
import pickle

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

# compute average block length (takes some time)
if False:
    t_length = np.array([[data[(data.subject == s) & (data.block == b)].t_length.sum() for b in range(nblocks)] for s in range(nsubjects)])
    print(f'Mean block length +- SEM: {np.mean(t_length):.1f} +- {sem(t_length.mean(axis=1)):.1f} (median = {np.median(t_length):.1f})')

# compute choice consistency in phase 1 per subject
if False:
    count, consistent = np.zeros(nsubjects), np.zeros(nsubjects)
    count2, consistent2 = np.zeros(nsubjects), np.zeros(nsubjects)
    count3, consistent3 = np.zeros(nsubjects), np.zeros(nsubjects)
    for s in range(nsubjects):
        print(f'Subject {s + 1} / {nsubjects}')
        for b in range(nblocks):
            d = data[(data.subject == s) & (data.block == b) & (data.phase == 1)]
            for p in d.pair.unique():
                if len(d[d.pair == p]) > 1:
                    trials = d[d.pair == p].trial_phase.values
                    for i, t in enumerate(trials[1:]):
                        count[s] += 1
                        consistent[s] += d[(d.pair == p) & (d.trial_phase == t)].choice.values[0] == d[(d.pair == p) & (d.trial_phase == trials[i])].choice.values[0]
                        if i == 0:
                            count2[s] += 1
                            consistent2[s] += d[(d.pair == p) & (d.trial_phase == t)].choice.values[0] == d[(d.pair == p) & (d.trial_phase == trials[i])].choice.values[0]
                        elif i == 1:
                            count3[s] += 1
                            consistent3[s] += d[(d.pair == p) & (d.trial_phase == t)].choice.values[0] == d[(d.pair == p) & (d.trial_phase == trials[i])].choice.values[0]

    pickle.dump((count, consistent, count2, consistent2, count3, consistent3), open('../results/behav/consistency.pkl', 'wb'))
else:
    count, consistent, count2, consistent2, count3, consistent3 = pickle.load(open('../results/behav/consistency.pkl', 'rb'))
r, p = pearsonr(data[data.phase == 0].groupby('subject').correct.mean().values, consistent/count)
print(f'Correlation between performance and consistency: r={r:.3f} (p={p:.5f})')

# compute absolute pairwise differences between ratings
if False:
    stim_combos = list(combinations(range(5), 2))
    count, absratingdiff1, absratingdiff2 = np.zeros(nsubjects), np.zeros(nsubjects), np.zeros(nsubjects)
    for s in range(nsubjects):
        print(f'Subject {s + 1} / {nsubjects}')
        for b in range(nblocks):
            d = data[(data.subject == s) & (data.block == b) & data.b_has_rating1 & data.b_has_rating2]
            if len(d):
                count[s] += 1
                absratingdiff1[s] += np.mean([np.abs(d[(d.stimulus_left == c[0]) & d.type_rating1].rating.values[0] - d[(d.stimulus_left == c[1]) & d.type_rating1].rating.values[0]) for c in stim_combos])
                absratingdiff2[s] += np.mean([np.abs(d[(d.stimulus_left == c[0]) & d.type_rating2].rating.values[0] - d[(d.stimulus_left == c[1]) & d.type_rating2].rating.values[0]) for c in stim_combos])
    pickle.dump((absratingdiff1 / count, absratingdiff2 / count), open('../results/behav/absratingdiff.pkl', 'wb'))
else:
    absratingdiff1, absratingdiff2 = pickle.load(open('../results/behav/absratingdiff.pkl', 'rb'))

stats = wilcoxon(absratingdiff1[np.setdiff1d(range(nsubjects), exclude)], absratingdiff2[np.setdiff1d(range(nsubjects), exclude)])
print(f'Absolute pairwise rating difference post vs. pre: W={stats.statistic:.1f} (p={stats.pvalue:.5f})')  # sign. if subject with perf<0.55 excluded


ps = ['trial_phase', 'b_designated_absvaluediff', 'b_valuebase', 'absvaluediff', 'b_stimulus_pool', 'value_chosen']
linear_regression(
    data[~data.confidence.isna() & (data.phase == 1) & data.type_choice & ~data.subject.isin(exclude)],
    patsy_string='confidence ~ ' + ' + '.join(ps) + ' + trial_phase*b_valuebase',
    standardize_vars=True,
    ignore_warnings=True,
    reml=False,
    print_data=False
)

ps = ['trial_phase', 'b_designated_absvaluediff', 'b_valuebase', 'absvaluediff', 'value_chosen']
linear_regression(
    data[~data.confidence.isna() & (data.phase == 1) & data.type_choice],
    patsy_string='confidence ~ ' + ' + '.join(ps),
    standardize_vars=True,
    ignore_warnings=True,
    reml=False,
    print_data=False
)

# compute confidence slopes and rating diffs across phase 1
if False:
    confslope, confslope2, confslope3, ratingdiff = np.full(nsubjects, np.nan), np.full(nsubjects, np.nan), np.full(nsubjects, np.nan), np.full(nsubjects, np.nan)
    for s in range(nsubjects):
        print(f'Subject {s + 1} / {nsubjects}')
        d = data[(data.subject == s) & data.b_has_rating1 & data.b_has_rating2]
        ratingdiff[s] = d[d.type_rating2].rating.mean() - d[d.type_rating1].rating.mean()

        confslope[s] = linregress(d[d.type_choice_noc & ~d.equal_value_pair].trial_phase.values,
                                  np.array(d[d.type_choice_noc & ~d.equal_value_pair].confidence.values, dtype=float)).slope
        confslope2[s] = linregress(data[(data.subject == s) & data.type_choice_noc & ~data.equal_value_pair].trial_phase.values,
                                   np.array(data[(data.subject == s) & data.type_choice_noc & ~data.equal_value_pair].confidence.values, dtype=float)).slope
        confslope3[s] = linregress(data[(data.subject == s) & (data.phase == 1)].trial_phase.values,
                                   np.array(data[(data.subject == s) & (data.phase == 1)].confidence.values, dtype=float)).slope
    pickle.dump(ratingdiff, open('../results/behav/ratingdiff.pkl', 'wb'))
    pickle.dump((confslope, confslope2, confslope3), open('../results/behav/confslope.pkl', 'wb'))
else:
    ratingdiff = pickle.load(open('../results/behav/ratingdiff.pkl', 'rb'))
    confslope, confslope2, confslope3 = pickle.load(open('../results/behav/confslope.pkl', 'rb'))
        # for b in range(nblocks):
r, p = pearsonr(confslope2, absratingdiff2-absratingdiff1)
print(f'Correlation between confidence slope and change in absolute rating difference: r={r:.3f} (p={p:.5f})')

if True:
    ps = ['trial_phase', 'b_designated_absvaluediff', 'b_valuebase', 'absvaluediff', 'valuesum', 'b_stimulus_pool', 'block']
    linear_regression(
        data[~data.confidence.isna() & (data.phase == 1) & data.type_choice],
        patsy_string='confidence ~ ' + ' + '.join(ps) + ' + b_valuebase:trial_phase',
        standardize_vars=True,
        ignore_warnings=True,
        model_blocks=False,
        reml=False,
        print_data=False
    )

# data.loc[data.b_ntrials_pre.isin([9, 12]), 'pre_short'] = 1
# data.loc[data.b_ntrials_pre.isin([15, 18]), 'pre_short'] = 0
ps = ['value_chosen', 'block', 'b_valuebase', 'b_designated_absvaluediff', 'b_stimulus_pool', 'b_ntrials_pre']
model = linear_regression(
    data[~data.ratingdiff21.isna()],
    # patsy_string='ratingdiff ~ ' + ' + '.join(ps),
    patsy_string='ratingdiff21 ~ ' + ' + '.join(ps) + ' + b_ntrials_pre*value_chosen*b_valuebase*b_ntrials_noc',
    # patsy_string='rating2 ~ ' + ' + '.join(ps) + ' + rating1:value_chosen',
    # patsy_string='rating2 ~ ' + ' + '.join(ps),
    standardize_vars=True,
    ignore_warnings=True,
    model_blocks=True,
    reml=False,
    print_data=False
)


ps = ['value_chosen']
model = linear_regression(
    data[~data.ratingdiff21.isna() & (data.b_ntrials_noc == 15)],
    patsy_string='ratingdiff21 ~ ' + ' + '.join(ps),
    # patsy_string='ratingdiff21 ~ ' + ' + '.join(ps) + ' + b_ntrials_pre*value_chosen*b_valuebase',
    # patsy_string='rating2 ~ ' + ' + '.join(ps) + ' + rating1:value_chosen',
    # patsy_string='rating2 ~ ' + ' + '.join(ps),
    standardize_vars=True,
    ignore_warnings=True,
    model_blocks=False,
    reml=False,
    print_data=False,
    silent=True
)


ps = ['value_chosen', 'block', 'b_valuebase', 'b_designated_absvaluediff', 'b_stimulus_pool', 'b_ntrials_noc', 'confslope']
linear_regression(
    data[~data.ratingdiff21.isna() & data.b_ntrials_pre.isin([9, 12])],
    # data[~data.ratingdiff21.isna() & data.b_ntrials_pre.isin([15, 18])],
    patsy_string='ratingdiff21 ~ ' + ' + '.join(ps),
    # patsy_string='ratingdiff ~ ' + ' + '.join(ps) + ' + confslope:value_chosen',
    # patsy_string='rating2 ~ ' + ' + '.join(ps) + ' + rating1:value_chosen',
    # patsy_string='rating2 ~ ' + ' + '.join(ps),
    standardize_vars=True,
    ignore_warnings=True,
    model_blocks=True,
    reml=False,
    print_data=False
)

# bd = data[]
ps = ['trial_phase', 'b_designated_absvaluediff', 'b_valuebase', 'absvaluediff', 'valuesum', 'b_stimulus_pool', 'block']
linear_regression(
    data[~data.confidence.isna() & (data.phase == 1) & data.type_choice],
    patsy_string='confidence ~ ' + ' + '.join(ps) + ' + b_valuebase:trial_phase',
    standardize_vars=True,
    ignore_warnings=True,
    model_blocks=False,
    reml=False,
    print_data=False
)