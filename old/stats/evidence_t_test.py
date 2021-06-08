import os
import numpy as np
import pandas as pd
import statsmodels.stats.weightstats as ssw

from pathlib import Path

cwd = Path.cwd()
path_data = os.path.join(cwd, '../results/')


model_1 = 3     # ConfGen
model_2 = 7     # ConfNoFeedGen

fittingData_1 = pd.read_pickle(os.path.join(path_data, 'fittingData/fittingDataM' + str(model_1) + '.pkl'))
fittingData_2 = pd.read_pickle(os.path.join(path_data, 'fittingData/fittingDataM' + str(model_2) + '.pkl'))

testing = ssw.ttost_paired(fittingData_1.AIC, fittingData_2.AIC, low=0.05, upp=0.95)
print('p = ' + str(testing[0]))
print('t1, pv1, df1 = ' + str(testing[1]))
print('t2, pv2, df2 = ' + str(testing[2]))
