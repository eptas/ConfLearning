import os
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from functools import reduce


cwd = Path.cwd()
path_data = os.path.join(cwd, '../../data/')

conf = np.load(os.path.join(path_data + 'confidence_value.npy'))

nsubjects = 66
ntrials = 15
nblocks = 10

confidence = np.full((nblocks, ntrials), np.nan)
confSlope = np.full(nsubjects, np.nan)


def slope_intercept(x, y):

    slope = (((np.mean(x) * np.mean(y)) - np.mean(x * y)) / ((np.mean(x) * np.mean(x)) - np.mean(x * x)))
    intercept = np.mean(y) - slope * np.mean(x)

    return slope, intercept


fig, axes = plt.subplots(11, 6, figsize=(20, 10))
rows = [*[0 for _ in range(6)], *[1 for _ in range(6)], *[2 for _ in range(6)], *[3 for _ in range(6)], *[4 for _ in range(6)], *[5 for _ in range(6)], *[6 for _ in range(6)], *[7 for _ in range(6)], *[8 for _ in range(6)], *[9 for _ in range(6)], *[10 for _ in range(6)]]
columns = reduce(operator.concat, [[0, 1, 2, 3, 4, 5] for _ in range(11)])


for s in range(nsubjects):

    for b in range(nblocks):
        index = np.where(~np.isnan(conf[s, b, 1, :]))
        confidence[b, :] = conf[s, b, 1, :][index[0][0]:(index[0][0] + ntrials)] if len(index[0]) >= 5 else np.full(ntrials, np.nan)

    confP1 = np.nanmean(confidence, axis=0)
    slope, intercept = slope_intercept(np.arange(ntrials), confP1)

    regLine = [(slope * trial) + intercept for trial in np.arange(ntrials)]
    confSlope[s] = slope

    row, col = rows[s], columns[s]

    axes[row, col].scatter(range(ntrials), confP1, s=8, c='g', marker='o')
    axes[row, col].plot(range(ntrials), regLine, color='r', linewidth=0.5)
    axes[row, col].set_title('s' + str(s))
    axes[max(rows), col].set_xlabel('trial')
    axes[row, 0].set_ylabel('conf')
    axes[row, col].set_xticks(np.arange(0, 10, step=2))
    axes[row, col].set_yticks(np.arange(0, 11, step=2))
    axes[row, col].text(5, 5, str(round(slope, 2)), color='k', fontsize=10)
    axes[row, col].grid('silver', linestyle='-', linewidth=0.4)

np.save('confSlope_5-15', confSlope)

fig.savefig('../../figures/validation/conf_slope/conf_slope_5-15.png', bbox_inches='tight')
plt.close()
