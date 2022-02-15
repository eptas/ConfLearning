import sys
import os
import matplotlib.pyplot as plt
from plot_model_conf import plot_MC
from plot_model_value import plot_value
from plot_model_EC import plot_EC
from plot_model_CPE import plot_CPE
from plot_model_ABSCPE import plot_ABSCPE
from plot_model_behavconf import plot_BC
import pandas as pd
import numpy as np
from scipy.stats import linregress, sem
import seaborn as sns
from confidence.ConfLearning.stats.regression import regression

# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))
from plot_util import set_fontsize, savefig  # noqa

colors = sns.color_palette()

fig = plt.figure(figsize=(8.5, 7.5))
gs = fig.add_gridspec(3, 15)



ax11 = fig.add_subplot(gs[0, :5])
handles_phase1, labels_phase1, handles_value, labels_value = \
plot_EC(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoSpec', title='Expected confidence')
# plt.text(-0.11, 1.04, 'D', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel(None)
plt.xticks(ax11.get_xticks(), [])
plt.ylabel('Expected confidence')
ax11.yaxis.set_label_coords(-0.13, 0.5)
plt.text(0.055, 1.2, 'A) ConfSpec', transform=ax11.transAxes, clip_on=False, fontsize=13, fontweight='bold', ha='center', fontstyle='italic')
# Plot legends
leg = plt.legend(handles_phase1, labels_phase1, loc='upper left', bbox_to_anchor=(-0.015, 1.04), title='No. trials\nin Phase 1:', fontsize=9, title_fontsize=9.5, labelspacing=-0.15, handlelength=2.5, frameon=False)
leg._legend_box.align = 'left'
ax11.add_artist(leg)
leg2 = plt.legend(handles_value[1:], labels_value, loc='upper left', bbox_to_anchor=(0.725, 1.058), fontsize=9, labelspacing=-0.3, handletextpad=2.5, framealpha=0.75, title='Stimulus:', title_fontsize=9.5, markerscale=0.5, borderpad=0.2)
ax11.add_artist(leg2)
plt.arrow(0.885, 0.74, 0, 0.19, color=(0.3, 0.3, 0.3), head_length=0.02, head_width=0.015, length_includes_head=True, clip_on=False, transform=ax11.transAxes, zorder=10, lw=0.75)
plt.text(0.955, 0.75, 'Value\nlevel', color=(0.3, 0.3, 0.3), ha='center', rotation=90, transform=ax11.transAxes, fontsize=9, zorder=10, linespacing=0.87)


ax12 = fig.add_subplot(gs[0, 5:10])
plot_CPE(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoSpec', title='Confidence prediction error')
# plt.text(-0.11, 1.04, 'A', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel(None)
plt.xticks(ax12.get_xticks(), [])
plt.ylabel('CPE')
ax12.yaxis.set_label_coords(-0.13, 0.5)
# plt.text(0.5, 1.25, 'B) Confidence', transform=ax22.transAxes, clip_on=False, fontsize=13, fontweight='bold', ha='center')
# ax12.yaxis.set_label_coords(-0.14, 0.5)

ax13 = fig.add_subplot(gs[0, 10:15])
plot_ABSCPE(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoSpec', title='Abs. confidence prediction error', title_xshift=-0.1)
# plt.text(-0.11, 1.04, 'A', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel(None)
plt.ylabel('Absolute CPE')
plt.xticks(ax13.get_xticks(), [])


ax21 = fig.add_subplot(gs[1, :5])
plot_value(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoSpec', title='Expected value')
# plt.text(-0.14, 1.04, 'E', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel(None)
plt.xticks(ax21.get_xticks(), [])
plt.ylabel('Value')
ax21.yaxis.set_label_coords(-0.12, 0.5)

ax22 = fig.add_subplot(gs[1, 5:10])
plot_MC(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoSpec', title='Confidence', plot_mean=False)
# plt.text(-0.14, 1.04, 'B', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel(None)
plt.xticks(ax22.get_xticks(), [])
plt.ylabel('Confidence')
ax22.yaxis.set_label_coords(-0.14, 0.5)

ax23 = fig.add_subplot(gs[1, 10:15])
BC = pd.read_pickle('MC_MonoSpec.pkl')
conf = np.array([[linregress(range(15), BC[(BC.subject == s) & (BC.phase == 1)].groupby('trial_phase')[f"MC{c}"].mean().values)[0] for s in range(66)] for c in range(4)])
df = pd.DataFrame(index=range(4*66))
df['confidence'], df['value'], df['subject'] = conf.flatten(), np.repeat(range(4), 66), np.tile(range(66), 4)
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
ax23.yaxis.set_label_coords(-0.22, 0.5)


ax31 = fig.add_subplot(gs[2, :5])
plot_value(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='Mono_choice', title='Expected value')
# plt.text(-0.14, 1.04, 'F', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.ylabel('Value')
ax31.yaxis.set_label_coords(-0.12, 0.5)
plt.text(0, 1.16, 'B) Choice', transform=ax31.transAxes, clip_on=False, fontsize=13, fontweight='bold', ha='center', fontstyle='italic')

ax32 = fig.add_subplot(gs[2, 5:10])
plot_MC(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='Mono_choice', title='Confidence', plot_mean=False)
# plt.text(-0.14, 1.04, 'C', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.ylabel('Confidence')
ax32.yaxis.set_label_coords(-0.14, 0.5)

ax33 = fig.add_subplot(gs[2, 10:15])
BC = pd.read_pickle('MC_Mono_choice.pkl')
conf = np.array([[linregress(range(15), BC[(BC.subject == s) & (BC.phase == 1)].groupby('trial_phase')[f"MC{c}"].mean().values)[0] for s in range(66)] for c in range(4)])
df = pd.DataFrame(index=range(4*66))
df['confidence'], df['value'], df['subject'] = conf.flatten(), np.repeat(range(4), 66), np.tile(range(66), 4)
reg = regression(df, 'confidence ~ value', silent=True, type='ols', standardize_vars=False)
for i in range(4):
    plt.bar(i, np.mean(conf, axis=1)[i], fc=colors[i], yerr=sem(conf, axis=1)[i], alpha=0.5, error_kw=dict(ecolor=colors[i], alpha=0.5))
plt.plot([0, 3], reg.params.Intercept + reg.params.value * np.array([0, 3]), 'k-', lw=2)
plt.text(0.075, 0.85, fr'$\beta={reg.params.value:.3f}$', transform=ax33.transAxes, fontsize=9)
plt.text(0.075, 0.75, fr'$p={reg.pvalues.value:.3f}$', transform=ax33.transAxes, fontsize=9)
plt.xticks(range(4), ['Lowest', '2nd\nlowest', '2nd\nhighest', 'Highest'], fontsize=8, linespacing=0.8)
plt.xlabel('CS value level')
plt.ylabel('Slope')
ax33.yaxis.set_label_coords(-0.22, 0.5)
plt.title('Confidence slopes')
plt.ylim(0, 0.0185)
import matplotlib.transforms
offset = matplotlib.transforms.ScaledTranslation(0, 3/72, plt.gcf().dpi_scale_trans)
for tick in ax33.xaxis.get_majorticklabels():
    tick.set_transform(tick.get_transform() + offset)

plt.subplots_adjust(left=0.06, right=0.99, top=0.925, bottom=0.095, hspace=0.275, wspace=20)
ax13.set_position(matplotlib.transforms.Bbox(ax13.get_position() + np.array([[0.0175, 0], [0, 0]])))
ax23.set_position(matplotlib.transforms.Bbox(ax23.get_position() + np.array([[0.0175, 0], [0, 0]])))
ax33.set_position(matplotlib.transforms.Bbox(ax33.get_position() + np.array([[0.0175, 0], [0, 0]])))
for ax in (ax31, ax32, ax33):
    ax.set_position(matplotlib.transforms.Bbox(ax.get_position() + np.array([[0, -0.03], [0, -0.03]])))
set_fontsize(label=11, tick=9)
savefig(f'../figures/model/compare_unspec_spec_choice_ext.png', pad_inches=0.01)
plt.show()