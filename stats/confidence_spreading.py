import numpy as np
import pandas as pd
from regression import regression
from scipy.stats import linregress

# modes = ('BC', 'MC_MonoUnspec', 'MC_MonoSpec', 'MC_Mono_choice', 'MC_Preservation')
modes = ('MC_Preservation',)

nsubjects = 66
nblocks = 11
include = np.setdiff1d(range(nsubjects), [25, 30])

for mode in modes:
    print(f'Mode: {mode}\n\n')
    data = pd.read_pickle(f'../plot/{mode}.pkl')
    conf = np.array([[linregress(range(15), data[(data.subject == s) & (data.phase == 1)].groupby('trial_phase')[f"{'BC' if mode == 'BC' else 'MC'}{c}"].mean().values)[0] for s in include] for c in range(4)])
    print(np.mean(conf, axis=1))
    df = pd.DataFrame(index=range(4*len(include)))
    df['confidence'] = conf.flatten()
    df['value'] = np.repeat(range(4), len(include))
    df['subject'] = np.tile(range(len(include)), 4)

    regression(df, 'confidence ~ value')
