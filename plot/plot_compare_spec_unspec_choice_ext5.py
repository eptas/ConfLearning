import sys
import os
import matplotlib.pyplot as plt
from util.plot_model_conf import plot_MC
from util.plot_model_value import plot_value
from util.plot_model_perf import plot_PERF
from util.plot_model_EC import plot_EC
from util.plot_model_CPE import plot_CPE
from util.plot_model_ABSCPE import plot_ABSCPE
from util.plot_model_behavconf import plot_BC
import pandas as pd
import numpy as np
from scipy.stats import linregress, sem
import seaborn as sns
from ConfLearning.stats.regression import regression
from pathlib import Path
import matplotlib.transforms
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))
from plot_util import set_fontsize, savefig  # noqa

cwd = Path.cwd()
path_data = os.path.join(cwd, '../data/')

colors = sns.color_palette()

fig = plt.figure(figsize=(9, 9))
gs = fig.add_gridspec(4, 19)

nsubjects = 66
include = np.setdiff1d(range(nsubjects), [25, 30])


ax11 = fig.add_subplot(gs[0, :5])
handles_phase1, labels_phase1, handles_value, labels_value = \
plot_EC(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoSpec', title='A) Expected confidence')
# plt.text(-0.11, 1.04, 'D', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel(None)
plt.xticks(ax11.get_xticks(), [])
# plt.ylabel('Expected confidence')
# ax11.yaxis.set_label_coords(-0.13, 0.5)
# Plot legends
leg = plt.legend(handles_phase1, labels_phase1, loc='upper left', bbox_to_anchor=(-0.019, 1.045), title='No. trials\nin Phase 1:', fontsize=9, title_fontsize=9.5, labelspacing=-0.175, handlelength=2.5, frameon=False)
leg._legend_box.align = 'left'
ax11.add_artist(leg)
leg2 = plt.legend(handles_value[1:], labels_value, loc='upper left', bbox_to_anchor=(0.625, 1.068), fontsize=9, labelspacing=-0.3, handletextpad=2.5, framealpha=0.75, title='Stimulus:', title_fontsize=9.5, markerscale=0.5, borderpad=0.2)
ax11.add_artist(leg2)
plt.arrow(0.782, 0.7425, 0, 0.2, color=(0.3, 0.3, 0.3), head_length=0.02, head_width=0.015, length_includes_head=True, clip_on=False, transform=ax11.transAxes, zorder=10, lw=0.75)
plt.text(0.86, 0.76, 'Value\nlevel', color=(0.3, 0.3, 0.3), ha='center', rotation=90, transform=ax11.transAxes, fontsize=9, zorder=10, linespacing=0.87)
# plt.text(0.01, 1.17, 'Model: $ConfSpec$', transform=ax11.transAxes, clip_on=False, fontsize=13, fontweight='bold', ha='center', fontstyle='italic')
plt.text(-0.14, 1.18, 'Model:', transform=ax11.transAxes, clip_on=False, fontsize=13, fontweight='bold')
plt.text(0.22, 1.18, 'ConfSpec', transform=ax11.transAxes, clip_on=False, fontsize=13, fontweight='bold', fontstyle='italic')

ax12 = fig.add_subplot(gs[0, 5:10])
plot_CPE(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoSpec', title='B) Confidence prediction error', title_xshift=-0.05)
# plt.text(-0.11, 1.04, 'A', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel(None)
plt.xticks(ax12.get_xticks(), [])
plt.yticks(ax12.get_yticks(), [f'{v:.1f}' for v in ax12.get_yticks()])
# plt.ylabel('CPE')
# ax12.yaxis.set_label_coords(-0.13, 0.5)
# plt.text(0.5, 1.25, 'B) Confidence', transform=ax22.transAxes, clip_on=False, fontsize=13, fontweight='bold', ha='center')
# ax12.yaxis.set_label_coords(-0.14, 0.5)
plt.ylim(-0.3, 0.55)

ax13 = fig.add_subplot(gs[0, 10:15])
plot_ABSCPE(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoSpec', title='C) Abs. confidence prediction error', title_xshift=0.05)
# plt.text(-0.11, 1.04, 'A', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel(None)
# plt.ylabel('Absolute CPE')
plt.xticks(ax13.get_xticks(), [])

ax14 = fig.add_subplot(gs[0, 15:19])
fit_spec = pd.read_pickle(os.path.join(path_data, "../results/fittingData/fittingData_MonoSpec_simchoice.pkl"))
alpha = fit_spec.ALPHA[np.setdiff1d(range(66), [25, 30])]
beta = fit_spec.BETA[np.setdiff1d(range(66), [25, 30])]
alpha_c = fit_spec.ALPHA_C[np.setdiff1d(range(66), [25, 30])]
gamma = np.log(fit_spec.GAMMA[np.setdiff1d(range(66), [25, 30])][fit_spec.GAMMA[np.setdiff1d(range(66), [25, 30])] != 0])
xlim = dict(alpha=(0, 1), beta=(0, np.round(beta.max())), gamma=(-0.5, np.ceil(gamma.max())), alpha_c=(0, 1))
ylim = dict(alpha=(0, 10), beta=(0, 20), gamma=(0, 12), alpha_c=(0, 32))
plt.hist(alpha_c, bins=32, color=(0.4, 0.4, 0.4))
plt.plot([alpha_c.mean(), alpha_c.mean()], ylim['alpha_c'], 'b-', lw=1.5)
plt.plot([np.median(alpha_c), np.median(alpha_c)], ylim['alpha_c'], 'g--', lw=1.5)
plt.xlim(xlim['alpha_c'])
plt.yticks(np.arange(0, 36, step=5))
plt.xticks(np.arange(0, 1.01, 0.2))
plt.ylim(ylim['alpha_c'])
plt.title(r'D) Histogram $\alpha_c$')
ax14.get_children()[0].set_color(np.array([170, 0, 0])/255)
inset_axes(ax14, width='100%', height='100%', bbox_to_anchor=(.35, .4, .55, .5), bbox_transform=ax14.transAxes)
plt.hist(alpha_c[alpha_c < 0.02], bins=18, color=np.array([170, 0, 0])/255)
plt.plot([np.median(alpha_c), np.median(alpha_c)], ylim['alpha_c'], 'g--', lw=1.5)
# plt.ylim(0, 25)
plt.xlim(0, 0.02)
plt.xticks(np.arange(0, 0.021, 0.02))
offset = matplotlib.transforms.ScaledTranslation(0, 3/72, plt.gcf().dpi_scale_trans)
for tick in ax14.xaxis.get_majorticklabels():
    tick.set_transform(tick.get_transform() + offset)

ax21 = fig.add_subplot(gs[1, :5])
plot_value(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoSpec', title='E) Expected value')
# plt.text(-0.14, 1.04, 'E', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel(None)
plt.xticks(ax21.get_xticks(), [])
# plt.ylabel('Value')
# ax21.yaxis.set_label_coords(-0.12, 0.5)

ax22 = fig.add_subplot(gs[1, 5:10])
plot_PERF(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoSpec', title='F) Performance')
# plt.text(-0.14, 1.04, 'E', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel(None)
plt.xticks(ax21.get_xticks(), [])
# plt.ylabel('Performance')
# ax21.yaxis.set_label_coords(-0.12, 0.5)

ax23 = fig.add_subplot(gs[1, 10:15])
plot_MC(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoSpec', title='G) Confidence', plot_mean=False)
# plt.text(-0.14, 1.04, 'B', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel(None)
plt.xticks(ax22.get_xticks(), [])
# plt.ylabel('Confidence')
# ax22.yaxis.set_label_coords(-0.14, 0.5)

ax24 = fig.add_subplot(gs[1, 15:19])
BC = pd.read_pickle('MC_MonoSpec.pkl')
conf = np.array([[linregress(range(15), BC[(BC.subject == s) & (BC.phase == 1)].groupby('trial_phase')[f"MC{c}"].mean().values)[0] for s in include] for c in range(4)])
df = pd.DataFrame(index=range(4*len(include)))
df['confidence'], df['value'], df['subject'] = conf.flatten(), np.repeat(range(4), len(include)), np.tile(range(len(include)), 4)
reg = regression(df, 'confidence ~ value', silent=True, type='ols', standardize_vars=False)
for i in range(4):
    plt.bar(i, np.mean(conf, axis=1)[i], fc=colors[i], yerr=sem(conf, axis=1)[i], alpha=0.5, error_kw=dict(ecolor=colors[i], alpha=0.5))
plt.plot([0, 3], reg.params.Intercept + reg.params.value * np.array([0, 3]), 'k-', lw=2)
plt.text(0.075, 0.85, fr'$\beta={reg.params.value:.3f}$', transform=ax24.transAxes, fontsize=9)
plt.text(0.075, 0.75, fr'$p={reg.pvalues.value:.3f}$', transform=ax24.transAxes, fontsize=9)
plt.xticks(range(4), [], fontsize=8)
plt.ylim(0, 0.0185)
tt = plt.title('H) Confidence slopes', y=0.98, x=0.4)
# tt.set_position((0.5, 0.5))
# plt.ylabel('Slope')
# ax23.yaxis.set_label_coords(-0.22, 0.5)


ax31 = fig.add_subplot(gs[2, :5])
plot_value(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='Mono_choice', title='I) Expected value')
# plt.text(-0.14, 1.04, 'F', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
# plt.ylabel('Value')
# ax31.yaxis.set_label_coords(-0.12, 0.5)
plt.xlabel(None)
plt.xticks(ax31.get_xticks(), [])
# plt.text(0.01, 1.18, 'Model: $Choice$', transform=ax31.transAxes, clip_on=False, fontsize=13, fontweight='bold', ha='center', fontstyle='italic')
plt.text(-0.14, 1.18, 'Model:', transform=ax31.transAxes, clip_on=False, fontsize=13, fontweight='bold')
plt.text(0.22, 1.18, 'Choice', transform=ax31.transAxes, clip_on=False, fontsize=13, fontweight='bold', fontstyle='italic')

ax32 = fig.add_subplot(gs[2, 5:10])
plot_PERF(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='Mono_choice', title='J) Performance')
# plt.text(-0.14, 1.04, 'C', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
# plt.ylabel('Confidence')
# ax32.yaxis.set_label_coords(-0.14, 0.5)
plt.xlabel(None)
plt.xticks(ax32.get_xticks(), [])

ax33 = fig.add_subplot(gs[2, 10:15])
plot_MC(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='Mono_choice', title='K) Confidence', plot_mean=False)
# plt.text(-0.14, 1.04, 'C', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
# plt.ylabel('Confidence')
# ax32.yaxis.set_label_coords(-0.14, 0.5)
plt.xlabel(None)
plt.xticks(ax33.get_xticks(), [])

ax34 = fig.add_subplot(gs[2, 15:19])
MC = pd.read_pickle('MC_Mono_choice.pkl')
conf = np.array([[linregress(range(15), MC[(MC.subject == s) & (MC.phase == 1)].groupby('trial_phase')[f"MC{c}"].mean().values)[0] for s in include] for c in range(4)])
df = pd.DataFrame(index=range(4*len(include)))
df['confidence'], df['value'], df['subject'] = conf.flatten(), np.repeat(range(4), len(include)), np.tile(range(len(include)), 4)
reg = regression(df, 'confidence ~ value', silent=True, type='ols', standardize_vars=False)
for i in range(4):
    plt.bar(i, np.mean(conf, axis=1)[i], fc=colors[i], yerr=sem(conf, axis=1)[i], alpha=0.5, error_kw=dict(ecolor=colors[i], alpha=0.5))
plt.plot([0, 3], reg.params.Intercept + reg.params.value * np.array([0, 3]), 'k-', lw=2)
plt.text(0.075, 0.85, fr'$\beta={reg.params.value:.3f}$', transform=ax34.transAxes, fontsize=9)
plt.text(0.075, 0.75, fr'$p={reg.pvalues.value:.3f}$', transform=ax34.transAxes, fontsize=9)
plt.xticks(range(4), [])
# plt.ylabel('Slope')
# ax33.yaxis.set_label_coords(-0.22, 0.5)
plt.title('L) Confidence slopes', x=0.4)
plt.ylim(0, 0.0185)


ax41 = fig.add_subplot(gs[3, :5])
plot_value(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='Perservation', title='M) Expected value')
# plt.text(-0.14, 1.04, 'F', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
# plt.ylabel('Value')
# ax31.yaxis.set_label_coords(-0.12, 0.5)
# plt.text(0.2, 1.18, r'Model: $\bm{Perservation}}$', transform=ax41.transAxes, clip_on=False, fontsize=13, fontweight='bold', ha='center', fontstyle='italic')
plt.text(-0.14, 1.18, 'Model:', transform=ax41.transAxes, clip_on=False, fontsize=13, fontweight='bold')
plt.text(0.22, 1.18, 'Perseveration', transform=ax41.transAxes, clip_on=False, fontsize=13, fontweight='bold', fontstyle='italic')
plt.xticks(ax41.get_xticks(), [str(v) for v in ax41.get_xticks()])

ax42 = fig.add_subplot(gs[3, 5:10])
plot_PERF(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='Perservation', title='N) Performance')
# plt.text(-0.14, 1.04, 'C', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
# plt.ylabel('Confidence')
# ax32.yaxis.set_label_coords(-0.14, 0.5)
plt.xticks(ax42.get_xticks(), [str(v) for v in ax42.get_xticks()])

ax43 = fig.add_subplot(gs[3, 10:15])
plot_MC(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='Perservation', title='O) Confidence', plot_mean=False)
# plt.text(-0.14, 1.04, 'C', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
# plt.ylabel('Confidence')
# ax32.yaxis.set_label_coords(-0.14, 0.5)
plt.xticks(ax43.get_xticks(), [str(v) for v in ax43.get_xticks()])

ax44 = fig.add_subplot(gs[3, 15:19])
MC = pd.read_pickle('MC_Perservation.pkl')
conf = np.array([[linregress(range(15), MC[(MC.subject == s) & (MC.phase == 1)].groupby('trial_phase')[f"MC{c}"].mean().values)[0] for s in include] for c in range(4)])
df = pd.DataFrame(index=range(4*len(include)))
df['confidence'], df['value'], df['subject'] = conf.flatten(), np.repeat(range(4), len(include)), np.tile(range(len(include)), 4)
reg = regression(df, 'confidence ~ value', silent=True, type='ols', standardize_vars=False)
for i in range(4):
    plt.bar(i, np.mean(conf, axis=1)[i], fc=colors[i], yerr=sem(conf, axis=1)[i], alpha=0.5, error_kw=dict(ecolor=colors[i], alpha=0.5))
plt.plot([0, 3], reg.params.Intercept + reg.params.value * np.array([0, 3]), 'k-', lw=2)
plt.text(0.075, 0.85, fr'$\beta={reg.params.value:.3f}$', transform=ax44.transAxes, fontsize=9)
plt.text(0.075, 0.75, fr'$p={reg.pvalues.value:.3f}$', transform=ax44.transAxes, fontsize=9)
plt.xticks(range(4), ['Lowest  ', '2nd\nlowest  ', '2nd\n  highest', '  Highest'], fontsize=8, linespacing=0.8)
plt.xlabel('CS value level')
# plt.ylabel('Slope')
# ax33.yaxis.set_label_coords(-0.22, 0.5)
plt.title('P) Confidence slopes', x=0.4)
plt.ylim(0, 0.0185)
offset = matplotlib.transforms.ScaledTranslation(0, 3/72, plt.gcf().dpi_scale_trans)
for tick in ax44.xaxis.get_majorticklabels():
    tick.set_transform(tick.get_transform() + offset)



# plt.tight_layout()
plt.subplots_adjust(top=0.94, bottom=0.06, hspace=0.4)

# first column
for ax in (ax11, ax21, ax31, ax41):
    ax.set_position(matplotlib.transforms.Bbox(ax.get_position() + np.array([[-0.095, 0], [-0.095, 0]])))
# second column
for ax in (ax12, ax22, ax32, ax42):
    ax.set_position(matplotlib.transforms.Bbox(ax.get_position() + np.array([[-0.038, 0], [-0.038, 0]])))
# third column
for ax in (ax13, ax23, ax33, ax43):
    ax.set_position(matplotlib.transforms.Bbox(ax.get_position() + np.array([[0.02, 0], [0.02, 0]])))
# last column
for ax in (ax14, ax24, ax34, ax44):
    ax.set_position(matplotlib.transforms.Bbox(ax.get_position() + np.array([[0.09, 0], [0.09, 0]])))
# first row
for ax in (ax11, ax12, ax13, ax14):
    ax.set_position(matplotlib.transforms.Bbox(ax.get_position() + np.array([[0, 0], [0, 0.015]])))
for ax in (ax11, ax12, ax13, ax14):
    ax.set_position(matplotlib.transforms.Bbox(ax.get_position() + np.array([[0, -0.01], [0, -0.015]])))
# second row
for ax in (ax21, ax22, ax23, ax24):
    ax.set_position(matplotlib.transforms.Bbox(ax.get_position() + np.array([[0, 0.01], [0, 0.01]])))
# last row
for ax in (ax41, ax42, ax43, ax44):
    ax.set_position(matplotlib.transforms.Bbox(ax.get_position() + np.array([[0, -0.01], [0, -0.01]])))
# all but last column
for ax in (ax11, ax12, ax13, ax21, ax22, ax23, ax31, ax32, ax33, ax41, ax42, ax43):
    ax.set_position(matplotlib.transforms.Bbox(ax.get_position() + np.array([[0, 0], [0.025, 0]])))
# all
for ax in (ax11, ax12, ax13, ax14, ax21, ax22, ax23, ax24, ax31, ax32, ax33, ax34, ax41, ax42, ax43, ax44):
    ax.set_position(matplotlib.transforms.Bbox(ax.get_position() + np.array([[0, 0], [0, 0.01]])))
ax24.set_position(matplotlib.transforms.Bbox(ax24.get_position() + np.array([[0, 0], [0, -0.005]])))

set_fontsize(label=11, tick=8, title=11)
savefig(f'../figures/model/compare_unspec_spec_choice_ext.png', pad_inches=0.01)
plt.show()