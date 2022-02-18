import sys
import os
import matplotlib.pyplot as plt
from util.plot_model_value import plot_value
from util.plot_model_EC import plot_EC
from util.plot_model_CPE import plot_CPE
from util.plot_model_conf import plot_MC
from util.plot_model_ABSCPE import plot_ABSCPE
from scipy.stats import linregress, sem
from histo_params_winning import plot_histo
import numpy as np
import pandas as pd
import seaborn as sns
from confidence.ConfLearning.stats.regression import regression

# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))
from plot_util import set_fontsize, savefig  # noqa

colors = sns.color_palette()

nsubjects = 66
include = np.setdiff1d(range(nsubjects), [25, 30])

fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(2, 16)

ax11 = fig.add_subplot(gs[0, :5])
plot_value(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False)
plt.text(-0.15, 1.04, 'A', transform=ax11.transAxes, color=(0, 0, 0), fontsize=17)

ax12 = fig.add_subplot(gs[0, 5:10])
plot_EC(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False)
plt.text(-0.14, 1.04, 'B', transform=ax12.transAxes, color=(0, 0, 0), fontsize=17)

ax13 = fig.add_subplot(gs[0, 10:15])
handles_phase1, labels_phase1, handles_value, labels_value = \
    plot_CPE(legend_value=False, legend_phase1=False, ylabel_as_title=True, markersize_value=6, reload=False)
plt.text(-0.14, 1.04, 'C', transform=ax13.transAxes, color=(0, 0, 0), fontsize=17)

# Plot legends
leg = plt.legend(handles_phase1, labels_phase1, loc='upper left', bbox_to_anchor=(1, 1.17), title='No. trials\nin Phase 1:', fontsize=9, title_fontsize=9.5, labelspacing=0.3, handlelength=4, frameon=False)
leg._legend_box.align = 'left'
ax13.add_artist(leg)
leg2 = plt.legend(handles_value, labels_value, loc='upper left', bbox_to_anchor=(1.02, 0.52), fontsize=9, labelspacing=0, handletextpad=2.5, framealpha=1, title='Stimulus:', title_fontsize=9.5)
ax13.add_artist(leg2)
plt.arrow(1.17, 0.06, 0, 0.3, color=(0.3, 0.3, 0.3), head_length=0.02, head_width=0.015, length_includes_head=True, clip_on=False, transform=plt.gca().transAxes, zorder=10, lw=0.75)
plt.text(1.24, 0.11, 'Value\nlevel', color=(0.3, 0.3, 0.3), ha='center', rotation=90, transform=ax13.transAxes, fontsize=9, zorder=10, linespacing=0.87)

ax21 = fig.add_subplot(gs[1, :5])
plot_ABSCPE(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False)
plt.text(-0.155, 1.04, 'D', transform=ax21.transAxes, color=(0, 0, 0), fontsize=17)


ax22 = fig.add_subplot(gs[1, 5:10])
plot_MC(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False)
plt.text(-0.16, 1.04, 'E', transform=ax22.transAxes, color=(0, 0, 0), fontsize=17)

ax23 = fig.add_subplot(gs[1, 10:15])
MC = pd.read_pickle('data/MC_MonoUnspec.pkl')
conf = np.array([[linregress(range(15), MC[(MC.subject == s) & (MC.phase == 1)].groupby('trial_phase')[f"MC{c}"].mean().values)[0] for s in include] for c in range(4)])
df = pd.DataFrame(index=range(4*len(include)))
df['confidence'], df['value'], df['subject'] = conf.flatten(), np.repeat(range(4), len(include)), np.tile(range(len(include)), 4)
reg = regression(df, 'confidence ~ value', silent=True, type='ols', standardize_vars=False)
for i in range(4):
    plt.bar(i, np.mean(conf, axis=1)[i], fc=colors[i], yerr=sem(conf, axis=1)[i], alpha=0.5, error_kw=dict(ecolor=colors[i], alpha=0.5))
plt.plot([0, 3], reg.params.Intercept + reg.params.value * np.array([0, 3]), 'k-', lw=2)
plt.text(0.075, 0.85, fr'$\beta={reg.params.value:.3f}$', transform=ax23.transAxes, fontsize=9)
plt.text(0.075, 0.75, fr'$p={reg.pvalues.value:.3f}$', transform=ax23.transAxes, fontsize=9)
plt.xticks(range(4), [], fontsize=8)
plt.title('Confidence slopes')
plt.ylim(0, 0.0185)
plt.ylabel('Slope')
# plt.text(0.5, 1.25, 'C) Confidence slope', transform=ax12.transAxes, clip_on=False, fontsize=13, fontweight='bold', ha='center')
# ax12.yaxis.set_label_coords(-0.22, 0.5)
plt.xlabel('CS value level')
plt.xticks(range(4), ['Lowest', '2nd\nlowest', '2nd\nhighest', 'Highest'], fontsize=8, linespacing=0.8)
import matplotlib.transforms
offset = matplotlib.transforms.ScaledTranslation(0, 3/72, plt.gcf().dpi_scale_trans)
for tick in ax23.xaxis.get_majorticklabels():
    tick.set_transform(tick.get_transform() + offset)
plt.text(-0.285, 1.04, 'F', transform=ax23.transAxes, color=(0, 0, 0), fontsize=17)

plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.11, hspace=0.5, wspace=3)
set_fontsize(label=11, tick=9)

ax11.set_position(matplotlib.transforms.Bbox(ax11.get_position() + np.array([[-0.01, 0], [-0.01, 0]])))
ax21.set_position(matplotlib.transforms.Bbox(ax21.get_position() + np.array([[-0.01, 0], [-0.01, 0]])))
ax23.set_position(matplotlib.transforms.Bbox(ax23.get_position() + np.array([[0.04, 0], [0.04, 0]])))

savefig(f'../figures/model/model_latentvars_new.png', pad_inches=0.01)
savefig(f'../figures/behav/Figure5.tif', pad_inches=0.01, format='tif', dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.show()