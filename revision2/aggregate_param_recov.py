import numpy as np
from glob import glob
import os

root = '/home/matteo/Dropbox/python/confidence/ConfLearning/results/'

modes = ('pearson', 'pearson_robust', 'spearman', 'slope_robust', 'negll', 'negll_true')

for m, mode in enumerate(modes):
    print(f'Mode {m + 1} / {len(modes)} [{mode}]')
    files = sorted(glob(os.path.join(root, 'param_recov', f'cm_{mode}_design*')))
    data = np.full((625, 4, (4, 250)[int('negll' in mode)]), np.nan)
    for i, file in enumerate(files):
        data[i] = np.load(file)
    np.save(os.path.join(root, f'cm2_{mode}.npy'), data)