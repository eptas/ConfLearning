import pandas as pd
import os
from scipy.stats import linregress
import numpy as np

data = pd.read_pickle('data.pkl')

nsubjects = 66
nblocks = 11
npairs = 10

for s in range(nsubjects):
    print(f'Subject {s + 1} / {nsubjects}')
    for b in range(nblocks):
        sbcond = (data.subject == s) & (data.block == b)
        cond = sbcond & data.b_has_rating1 & data.b_has_rating2 & data.type_rating
        if len(data[cond]):
            data.loc[cond & data.type_rating2, 'ratingdiff21'] = data[cond & data.type_rating2].rating.values - [data[cond & data.type_rating1 & (data.stimulus_left == st)].rating.values[0] for st in data[cond & data.type_rating2].stimulus_left]
            data.loc[cond & data.type_rating2, 'rating1'] = [data[cond & data.type_rating1 & (data.stimulus_left == st)].rating.values[0] for st in data[cond & data.type_rating2].stimulus_left]
            data.loc[cond & data.type_rating2, 'rating2'] = data[cond & data.type_rating2].rating.values
            data.loc[sbcond, 'ratingdiff_av'] = data[cond & data.type_rating2].rating.mean() - data[cond & data.type_rating1].rating.mean()
            data.loc[sbcond, 'confslope'] = linregress(data[sbcond & data.type_choice_noc & ~data.equal_value_pair].trial_phase.values,
                                                       np.array(data[sbcond & data.type_choice_noc & ~data.equal_value_pair].confidence.values, dtype=float)).slope

data.to_pickle('data.pkl')