import os
import csv
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from pathlib import Path

# for different methods refer to: https://www.statsmodels.org/stable/regression.html
# os.makedirs('../../results/validation')

cwd = Path.cwd()

path_data_d = os.path.join(cwd, '../../data/')
path_data_r = os.path.join(cwd, '../../results/fittingData')

df = pd.read_pickle(os.path.join(path_data_d, 'data.pkl'))
conf_slope = np.load('confSlope.npy')

nsubjects = 66
modellist = np.arange(2, 10)

rating_inc = np.zeros(nsubjects)

for s in range(nsubjects):
    rating_inc[s] = df[(df.subject == s) & df.type_rating2].rating.mean() - df[(df.subject == s) & df.type_rating1].rating.mean()

for m, model in enumerate(modellist):

    fittingData = pd.read_pickle(os.path.join(path_data_r, 'fittingDataM' + str(model) + '.pkl'))
    gamma = fittingData.GAMMA

    values = pd.DataFrame(data={"rating_inc": rating_inc, "conf_slope": conf_slope, "gamma": gamma}, columns=["rating_inc", "conf_slope", "gamma"])

    values = (values - values.mean()) / values.std()

    res = smf.ols(formula='rating_inc ~ conf_slope + gamma + conf_slope:gamma', data=values).fit()
    results = res.summary()

    results_text = results.as_text()

    with open(os.path.join(cwd, '../../results/validation/regM' + str(model) + '.csv'), 'w') as resultFile:

        resultFile.write(results_text)
        resultFile.close()
