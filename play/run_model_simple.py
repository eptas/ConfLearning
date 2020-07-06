import numpy as np
import os
from pathlib import Path
import pandas as pd

cwd = Path.cwd()
path_data = os.path.join(cwd, '../data/')

nblocks = 11
nphases = 3
nbandits = 5
ntrials = 56
ntrials_phase_max = 18
matrix = pd.read_pickle(os.path.join(path_data, 'data.pkl'))
stim_left = np.load(os.path.join(path_data, 'stim_left.npy'))
stim_right = np.load(os.path.join(path_data, 'stim_right.npy'))
chosen_stim = np.load(os.path.join(path_data, 'chosen_stim.npy'))
outcome_value = np.load(os.path.join(path_data, 'outcome_value.npy'))
confidence_value = np.load(os.path.join(path_data, 'confidence_value.npy'))
true_value = np.load(os.path.join(path_data, 'true_value.npy'))
correct_value = np.load(os.path.join(path_data, 'correct_value.npy'))


