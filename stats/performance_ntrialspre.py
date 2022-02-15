import numpy as np
import pandas as pd
from regression import regression
from scipy.stats import linregress
import matplotlib.pyplot as plt

# modes = ('BC', 'MC_MonoUnspec', 'MC_MonoSpec')
modes = ('BP', 'PERF_MonoUnspec', 'PERF_MonoSpec', 'PERF_Mono_choice', 'PERF_BetaSlope', 'PERF_Preservation')
# modes = ('PERF_Static', )

ntrials_phase0 = (9, 12, 15, 18)

nsubjects = 66
nblocks = 11
include = np.setdiff1d(range(nsubjects), [25, 30])

# plt.figure(figsize=(5, 3.5))

for m, mode in enumerate(modes):

    data = pd.read_pickle(f'../plot/{mode}.pkl')
    perf = np.array([[np.nan if len(data[(data.subject == s) & (data.phase == 1) & (data.b_ntrials_pre == nt) & data.type_choice & ~data.equal_value_pair].groupby('trial_phase')[f"{'BP' if mode == 'BP' else 'PERF'}"].mean().values) == 0 else linregress(range(len(data[(data.subject == s) & (data.phase == 1) & (data.b_ntrials_pre == nt) & data.type_choice & ~data.equal_value_pair].groupby('trial_phase')[f"{'BP' if mode == 'BP' else 'PERF'}"].mean().values)), data[(data.subject == s) & (data.phase == 1) & (data.b_ntrials_pre == nt) & data.type_choice & ~data.equal_value_pair].groupby('trial_phase')[f"{'BP' if mode == 'BP' else 'PERF'}"].mean().values)[0] for s in include] for nt in ntrials_phase0])
    # perf = np.array([[linregress(range(len(data[(data.subject == s) & (data.phase == 1)].groupby('trial_phase')[f"{'BP' if mode == 'BP' else 'PERF'}"].mean().values)), data[(data.subject == s) & (data.phase == 1) & (data.b_ntrials_pre == nt)].groupby('trial_phase')[f"{'BP' if mode == 'BP' else 'PERF'}"].mean().values)[0] for s in include] for nt in ntrials_phase0])
    print(f'{mode}: {np.nanmean(perf, axis=1)}')

    df = pd.DataFrame(index=range(4*len(include)))
    df['performance'] = perf.flatten()
    df['ntrials_pre'] = np.repeat(ntrials_phase0, len(include))
    df['subject'] = np.tile(range(len(include)), 4)

    for i in df[df.performance.isna()].index:
        df.loc[df.index == i, 'performance'] = df[df.ntrials_pre == df.iloc[i].ntrials_pre].performance.mean()

    reg = regression(df, 'performance ~ ntrials_pre', standardize_vars=False)

    # plt.subplot(2, 2, m + 1)
