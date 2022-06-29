import os
import sys
import warnings

import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import sem, linregress
import matplotlib.pyplot as plt
import seaborn as sns
from util.plot_model_behavconf import plot_BC

# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))
from ConfLearning.plot.util.plot_util import set_fontsize, savefig  # noqa
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from stats.regression import regression # noqa

path_data = os.path.join(Path.cwd(), '../data/')

data = pd.read_pickle(os.path.join(path_data, 'data.pkl'))

ntrials_phase0 = (9, 12, 15, 18)
ntrials_phase1 = (0, 5, 10, 15)
ntrials_phase2 = (9, 12, 15, 18)
nt_phase1_max = np.max(ntrials_phase1)
nt_phase2_max = np.max(ntrials_phase2)
nt_phase0phase1 = 27

colors = sns.color_palette()


nsubjects = 66
include = np.setdiff1d(range(nsubjects), [25, 30])
data = data[data.type_choice & ~data.equal_value_pair & data.subject.isin(include)]


window = 4

data['confidence'] /= 10

fig = plt.figure(figsize=(11, 4))
gs = fig.add_gridspec(1, 11)

linestyles = [(0, (1, 0.5)), (0, (4, 1)), '-.', '-']

ax = fig.add_subplot(gs[0, :4])

for i, nt in enumerate(ntrials_phase0):
    d0 = data[(data.b_ntrials_pre == nt) & (data.phase == 0)].groupby(['subject', 'trial_phase_rev']).correct.mean()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        m = d0.mean(level='trial_phase_rev').rolling(window=window, center=True).mean().values.astype(float)
        se = d0.sem(level='trial_phase_rev').rolling(window=window, center=True).mean().values.astype(float)
    plt.plot(np.arange(-nt+1.5, 1.5), m, lw=2, color='grey', alpha=0.6, ls=linestyles[i])
    plt.fill_between(np.arange(-nt+1.5, 1.5), m-se/2, m+se/2, lw=0, color='grey', alpha=0.4)

plt.axvspan(1, nt_phase1_max, facecolor='0.9', alpha=0.5, zorder=-11)
# plt.axhspan(0, 0.5, facecolor='0.85', alpha=0.5)
for i, nt in enumerate(ntrials_phase0):
    d1 = data[(data.b_ntrials_pre == nt) & (data.phase == 1)].groupby(['subject', 'trial_phase']).correct.mean()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        m = d1.mean(level='trial_phase').rolling(window=window, center=True).mean().values.astype(float)
        se = d1.sem(level='trial_phase').rolling(window=window, center=True).mean().values.astype(float)
    plt.plot(np.arange(0.5, nt_phase1_max+0.5), m, lw=2, color='grey', alpha=0.6, ls=linestyles[i])
    plt.fill_between(np.arange(0.5, nt_phase1_max+0.5), m-se/2, m+se/2, lw=0, color='grey', alpha=0.4)
d1 = data[(data.phase == 1)].groupby(['subject', 'trial_phase']).correct.mean()
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    m = d1.mean(level='trial_phase').rolling(window=window, center=True).mean().values.astype(float)
plt.plot(np.arange(0.5, nt_phase1_max+0.5), m, lw=3, color='k', alpha=0.6)

for i, nt in enumerate(ntrials_phase0):
    d2 = data[(data.b_ntrials_pre == nt) & (data.phase == 2)].groupby(['subject', 'trial_phase']).correct.mean()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        m = d2.mean(level='trial_phase').rolling(window=window, center=True).mean().values.astype(float)
        se = d2.sem(level='trial_phase').rolling(window=window, center=True).mean().values.astype(float)
    plt.plot(np.arange(nt_phase1_max-0.5, nt_phase1_max+nt_phase0phase1-nt-1.5), m, lw=2, color='grey', alpha=0.6, ls=linestyles[i], label=ntrials_phase0[i])
    plt.fill_between(np.arange(nt_phase1_max-0.5, nt_phase1_max+nt_phase0phase1-nt-1.5), m-se/2, m+se/2, lw=0, color='grey', alpha=0.4)
plt.plot([-20, 35], [0.5, 0.5], 'k-', lw=0.5, zorder=-10)
y_text = 0.46
plt.yticks(np.arange(0.5, 1.01, 0.1))
plt.ylim(0.45, 1)
plt.ylabel('Proportion correct')
plt.xlim(-20, 35)
plt.xticks(np.arange(-20, 40, 5))
plt.text(-10, y_text, 'Phase 1', ha='center', fontsize=11)
plt.text(8, y_text, 'Phase 2', ha='center', fontsize=11)
plt.text(25, y_text, 'Phase 3', ha='center', fontsize=11)
plt.xlabel('Trial')
plt.text(-0.2, 0.97, 'A', transform=ax.transAxes, color=(0, 0, 0), fontsize=20)

# handles_phase1, labels_phase1 = handles_linestyle[::-1], ntrials_phase0[::-1]
handles_phase1, labels_phase1 = ax.get_legend_handles_labels()

leg = plt.legend(handles_phase1[::-1], labels_phase1[::-1], loc='upper left', bbox_to_anchor=(0.7, 0.5), title='No. trials\nin Phase 1', fontsize=9, title_fontsize=9.5, labelspacing=0.5, handlelength=4, frameon=False)
# leg = plt.legend(loc='upper left', title='No. trials in Phase 1', fontsize=9, title_fontsize=9.5, labelspacing=0.5, handlelength=4, frameon=False)
leg._legend_box.align = 'left'
plt.gca().add_artist(leg)


ax2 = fig.add_subplot(gs[0, 4:8])
handles_phase1, labels_phase1, handles_value, labels_value = \
plot_BC(legend_value=False, legend_phase1=True, ylabel_as_title=False, reload=False, plot_mean=True, plot_value_levels=True)
# plt.text(-0.11, 1.04, 'A', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel('Trial')
plt.ylabel('Confidence')
# plt.xticks(ax2.get_xticks(), [])
plt.text(-0.2, 0.97, 'B', transform=ax2.transAxes, color=(0, 0, 0), fontsize=20)

# Plot legends
# leg = plt.legend(handles_phase1, labels_phase1, loc='upper left', bbox_to_anchor=(0.65, 0.5), title='No. trials\nin Phase 1:', fontsize=9, title_fontsize=9.5, labelspacing=0.3, handlelength=4, frameon=False)
# leg._legend_box.align = 'left'
# ax2.add_artist(leg)
leg2 = plt.legend(handles_value, labels_value, loc='upper left', bbox_to_anchor=(0.45, 1), fontsize=9, labelspacing=0.1, handletextpad=2.5, framealpha=1, title='Stimulus:', title_fontsize=9.5)
plt.gca().add_artist(leg2)
plt.arrow(0.585, 0.725, 0, 0.19, color=(0.3, 0.3, 0.3), head_length=0.02, head_width=0.015, length_includes_head=True, clip_on=False, transform=plt.gca().transAxes, zorder=10, lw=0.75)
plt.text(0.645, 0.77, 'Value\nlevel', color=(0.3, 0.3, 0.3), ha='center', rotation=90, transform=plt.gca().transAxes, fontsize=9, zorder=10, linespacing=0.87)



ax3 = fig.add_subplot(gs[0, 8:11])
BC = pd.read_pickle('data/BC.pkl')
conf = np.array([[linregress(range(15), BC[(BC.subject == s) & (BC.phase == 1)].groupby('trial_phase')[f"BC{c}"].mean().values)[0] for s in include] for c in range(4)])
df = pd.DataFrame(index=range(4*len(include)))
df['confidence'], df['value'], df['subject'] = conf.flatten(), np.repeat(range(4), len(include)), np.tile(range(len(include)), 4)
reg = regression(df, 'confidence ~ value', silent=True, type='ols', standardize_vars=False)
for i in range(4):
    plt.bar(i, np.mean(conf, axis=1)[i], fc=colors[i], yerr=sem(conf, axis=1)[i], alpha=0.5, error_kw=dict(ecolor=colors[i], alpha=0.5))
plt.plot([0, 3], reg.params.Intercept + reg.params.value * np.array([0, 3]), 'k-', lw=2)
plt.text(0.075, 0.5, fr'$\beta={reg.params.value:.3f}$', transform=ax3.transAxes, fontsize=11)
plt.text(0.075, 0.43, fr'$p={reg.pvalues.value:.3f}$', transform=ax3.transAxes, fontsize=11)
plt.xticks(range(4), [])
plt.yticks(np.arange(0, 0.016, 0.005))
# plt.title('Behavior')
# plt.text(0.5, 1.25, 'C) Confidence slope', transform=ax3.transAxes, clip_on=False, fontsize=13, fontweight='bold', ha='center')
plt.ylim(0, 0.0185)
plt.ylabel('Confidence slope')
# ax3.yaxis.set_label_coords(-0.22, 0.5)
plt.text(-0.37, 0.97, 'C', transform=ax3.transAxes, color=(0, 0, 0), fontsize=20)
plt.xlabel('CS value level')
plt.xticks(range(4), ['Lowest', '2nd\nlowest', '2nd\nhighest', 'Highest'], fontsize=8, linespacing=0.8)
import matplotlib.transforms
offset = matplotlib.transforms.ScaledTranslation(0, 3/72, plt.gcf().dpi_scale_trans)
for tick in ax3.xaxis.get_majorticklabels():
    tick.set_transform(tick.get_transform() + offset)

set_fontsize(label=12, tick=10)
plt.subplots_adjust(left=0.06, right=0.99, wspace=7, bottom=0.14)
ax2.set_position(matplotlib.transforms.Bbox(ax2.get_position() + np.array([[-0.008, 0], [-0.008, 0]])))
# savefig('../figures/behav/perf_conf_over_trials.png')
savefig(f'../figures/Figure2.tif', format='tif', dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.show()