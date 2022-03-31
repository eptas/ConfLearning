import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from pathlib import Path
sys.path.append(os.path.dirname(__file__))      # This is a trick to import local packages (without Pycharm complaining)
# from plot_util import set_fontsize, savefig  # noqa


model_list = ['RescorlaChoiceMono', 'RescorlaConfBase', 'RescorlaConfBaseGen']
model_id = 0
ndatasets = 100

ntrials_phase0 = (9, 12, 15, 18)
ntrials_phase1 = (0, 5, 10, 15)
ntrials_phase2 = (9, 12, 15, 18)
nt_phase1_max = np.max(ntrials_phase1)
nt_phase2_max = np.max(ntrials_phase2)
nt_phase0phase1 = 27
nblocks = 11

colors = sns.color_palette()

sim_choice = np.load('simu_choices_' + model_list[model_id] + '.npy')
sim_outcome = np.load('simu_outcomes_' + model_list[model_id] + '.npy')
sim_confidence = np.load('simu_confidences_' + model_list[model_id] + '.npy')

value_index = np.load('value_id_mani_10.npy')
values = sorted(np.unique(value_index[~np.isnan(value_index)]))
nvalues = len(values)

pairs = np.load('pair_mani_10.npy')
unique_pairs = np.unique(pairs)

trial_phase = np.load('trial_phase_mani_10.npy')


reload = True

if reload:

    count, consistent = np.zeros(ndatasets), np.zeros(ndatasets)
    count2, consistent2 = np.zeros(ndatasets), np.zeros(ndatasets)
    count3, consistent3 = np.zeros(ndatasets), np.zeros(ndatasets)

    for s in range(ndatasets):
        print(f'Subject {s + 1} / {ndatasets}')

        for b in range(nblocks):
            # d = data[(data.subject == s) & (data.block == b) & (data.phase == 1)]
            for p in unique_pairs:      # d.pair.unique():

                trial_length = len(sim_choice[s, 0, 0, b, 1, :][~np.isnan(sim_choice[s, 0, 0, b, 1, :])])

                var = [x for x, i in enumerate(pairs[b, 1, :][x]) if
                       len(pairs[b, 1, :][~np.isnan(pairs[b, 1, :])][x]) == trial_length]

                ##   A D D   L I N E  :  choices-array muss die gleiche Länge haben, wie trials >> basierend darauf array [0, 1, 2] auswählen

                pair_index = list(np.where(pairs[b, 1, :].flatten() == p)[0])

                if len(pair_index) > 1:

                # if len(d[d.pair == p]) > 1:             # CHANGE HERE WITH INDEX OF PAIRS !!!

                    trials = [int(trial_phase[b, 1, :].flatten()[x]) for x in pair_index]       # d[d.pair == p].trial_phase.values

                    for i, t in enumerate(trials[1:]):

                        count[s] += 1

                        consistency = d[(d.pair == p) & (d.trial_phase == t)].choice.values[0] == d[(d.pair == p) & (d.trial_phase == trials[i])].choice.values[0]
                        consistent[s] += consistency
                        data.loc[(data.subject == s) & (data.block == b) & (data.phase == 1) & (data.trial_phase == t), 'repeat_nr'] = i
                        data.loc[(data.subject == s) & (data.block == b) & (data.phase == 1) & (data.trial_phase == t), 'consistent'] = int(consistency)

                        if i == 0:
                            count2[s] += 1
                            consistent2[s] += consistency
                        elif i == 1:
                            count3[s] += 1
                            consistent3[s] += consistency

    pickle.dump((count, consistent, count2, consistent2, count3, consistent3), open('consistency.pkl', 'wb'))
else:
    count, consistent, count2, consistent2, count3, consistent3 = pickle.load(open('consistency.pkl', 'rb'))

count, consistent = count[np.setdiff1d(range(ndatasets), [25, 30])], consistent[np.setdiff1d(range(ndatasets), [25, 30])]
count2, consistent2 = count2[np.setdiff1d(range(ndatasets), [25, 30])], consistent2[np.setdiff1d(range(ndatasets), [25, 30])]
count3, consistent3 = count3[np.setdiff1d(range(ndatasets), [25, 30])], consistent3[np.setdiff1d(range(ndatasets), [25, 30])]

plt.figure(figsize=(7, 3))

ax1 = plt.subplot(121)
plt.hist(consistent2 / count2, bins=np.arange(0, 1.01, 0.02), facecolor=colors[0], alpha=0.7, label=r'1st $\rightarrow$ 2nd choice', zorder=6)
plt.hist(consistent3 / count3, bins=np.arange(0, 1.01, 0.02), facecolor=colors[1], alpha=0.7, label=r'2nd $\rightarrow$ 3rd choice', zorder=5)

plt.xlabel('Consistency')
plt.ylabel('Number of participants')
plt.xlim(0.5, 1)
plt.legend(loc='upper left')
plt.text(-0.3, 0.97, 'A', transform=ax1.transAxes, color=(0, 0, 0), fontsize=20)


ax2 = plt.subplot(122)
ratingdiff21 = data.groupby(['subject', 'value_id']).ratingdiff21.mean().groupby(level='value_id').mean()
ratingdiff21_se = data.groupby(['subject', 'value_id']).ratingdiff21.mean().groupby(level='value_id').sem()

for i, v in enumerate(values):
    plt.bar(i, ratingdiff21[i], yerr=ratingdiff21_se[i], facecolor=colors[i])

plt.plot([-0.75, nvalues-0.25], [0, 0], 'k-', lw=0.5)
plt.xticks(range(nvalues), ['Lowest', '2nd\nlowest', '2nd\nhighest', 'Highest'], fontsize=9)
plt.yticks(np.arange(-0.02, 0.021, 0.01))
plt.xlabel('CS value level')
plt.ylabel('Rating change (post - pre)')
plt.xlim(-0.5, nvalues-0.5)
plt.ylim(-0.02, 0.02)
plt.text(-0.43, 0.97, 'B', transform=ax2.transAxes, color=(0, 0, 0), fontsize=20)

plt.set_fontsize(label=11, tick=10)
plt.tight_layout()
plt.subplots_adjust(wspace=0.5, left=0.11)
plt.savefig(f'../figures/behav/Figure3.tif', format='tif', dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.show()
