import sys
import os
import matplotlib.pyplot as plt
from plot_model_EC import plot_EC
from plot_model_CPE import plot_CPE
from plot_model_ABSCPE import plot_ABSCPE
from plot_model_value import plot_value
from pathlib import Path
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.transforms

# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))
from plot_util import set_fontsize, savefig  # noqa

cwd = Path.cwd()
path_data = os.path.join(cwd, '../data/')

fig = plt.figure(figsize=(8, 7.5))
gs = fig.add_gridspec(4, 22)

ax11 = fig.add_subplot(gs[0, :10])
plot_EC(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoUnspec', title='Expected confidence')
# plt.text(-0.11, 1.04, 'A', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel(None)
plt.xticks(ax11.get_xticks(), [])
plt.text(10, 0.65, 'ConfUnspec', clip_on=False, fontsize=13, fontweight='bold', ha='center', fontstyle='italic')

ax12 = fig.add_subplot(gs[0, 10:20])
handles_phase1, labels_phase1, handles_value, labels_value = \
plot_EC(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoSpec', title='Expected confidence')
# plt.text(-0.14, 1.04, 'B', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel(None)
plt.xticks(ax11.get_xticks(), [])
plt.text(10, 0.65, 'ConfSpec', clip_on=False, fontsize=13, fontweight='bold', ha='center', fontstyle='italic')

# Plot legends
leg = plt.legend(handles_phase1, labels_phase1, loc='upper left', bbox_to_anchor=(1, 1.3), title='No. trials\nin Phase 1:', fontsize=9, title_fontsize=9.5, labelspacing=0.3, handlelength=4, frameon=False)
leg._legend_box.align = 'left'
ax12.add_artist(leg)
leg2 = plt.legend(handles_value[1:], labels_value, loc='upper left', bbox_to_anchor=(1.02, 0.53), fontsize=9, labelspacing=0.1, handletextpad=2.5, framealpha=1, title='Stimulus:', title_fontsize=9.5)
plt.gca().add_artist(leg2)
plt.arrow(1.15, -0.05, 0, 0.4, color=(0.3, 0.3, 0.3), head_length=0.02, head_width=0.015, length_includes_head=True, clip_on=False, transform=plt.gca().transAxes, zorder=10, lw=0.75)
plt.text(1.21, 0.06, 'Value\nlevel', color=(0.3, 0.3, 0.3), ha='center', rotation=90, transform=plt.gca().transAxes, fontsize=9, zorder=10, linespacing=0.87)


ax21 = fig.add_subplot(gs[1, :10])
plot_CPE(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoUnspec', title='Confidence prediction error')
# plt.text(-0.11, 1.04, 'C', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel(None)
plt.xticks(ax11.get_xticks(), [])

ax22 = fig.add_subplot(gs[1, 10:20])
plot_CPE(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoSpec', title='Confidence prediction error')
# plt.text(-0.14, 1.04, 'D', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel(None)
plt.xticks(ax11.get_xticks(), [])

ax31 = fig.add_subplot(gs[2, :10])
plot_ABSCPE(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoUnspec', title='Abs. confidence prediction error')
# plt.text(-0.11, 1.04, 'E', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)

ax32 = fig.add_subplot(gs[2, 10:20])
plot_ABSCPE(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoSpec', title='Abs. confidence prediction error')
# plt.text(-0.14, 1.04, 'F', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)

ax41 = fig.add_subplot(gs[3, :5])
fit_unspec = pd.read_pickle(os.path.join(path_data, "../results/fittingData/fittingData_MonoUnspec_simchoice.pkl"))
alpha_c = fit_unspec.ALPHA_C[np.setdiff1d(range(66), [25, 30])]
gamma = np.log(fit_unspec.GAMMA[np.setdiff1d(range(66), [25, 30])][fit_unspec.GAMMA[np.setdiff1d(range(66), [25, 30])] != 0])
plt.hist(alpha_c, bins=32, color=(0.4, 0.4, 0.4))
plt.title(r'Histogram $\alpha_c$')
plt.xticks(np.arange(0, 1.1, 0.2))
plt.xlim((0, 1))
plt.yticks(np.arange(0, 36, step=5))
plt.ylim(0, 37)
ax41.get_children()[0].set_color(np.array([170, 0, 0])/255)
inset_axes(ax41, width='100%', height='100%', bbox_to_anchor=(.35, .4, .55, .5), bbox_transform=ax41.transAxes)
plt.hist(alpha_c[alpha_c < 0.02], bins=18, color=np.array([170, 0, 0])/255)
# plt.ylim(0, 25)
plt.xlim(0, 0.02)
plt.xticks(np.arange(0, 0.021, 0.02))
ax42 = fig.add_subplot(gs[3, 5:10])
plt.hist(gamma, bins=32, color=(0.4, 0.4, 0.4))
plt.title(r'Histogram $\gamma$')
plt.xticks(np.arange(0, 5))
plt.yticks(np.arange(0, 36, step=5))
plt.ylim(0, 37)
plt.xlim(0, 4)

ax43 = fig.add_subplot(gs[3, 10:15])
fit_spec = pd.read_pickle(os.path.join(path_data, "../results/fittingData/fittingData_MonoSpec_simchoice.pkl"))
alpha_c = fit_spec.ALPHA_C[np.setdiff1d(range(66), [25, 30])]
gamma = np.log(fit_spec.GAMMA[np.setdiff1d(range(66), [25, 30])][fit_spec.GAMMA[np.setdiff1d(range(66), [25, 30])] != 0])
plt.hist(alpha_c, bins=32, color=(0.4, 0.4, 0.4))
plt.title(r'Histogram $\alpha_c$')
plt.xticks(np.arange(0, 1.1, 0.2))
plt.xlim((0, 1))
plt.yticks(np.arange(0, 36, step=5))
plt.ylim(0, 37)
ax43.get_children()[0].set_color(np.array([170, 0, 0])/255)
inset_axes(ax43, width='100%', height='100%', bbox_to_anchor=(.35, .4, .55, .5), bbox_transform=ax43.transAxes)
plt.hist(alpha_c[alpha_c < 0.02], bins=18, color=np.array([170, 0, 0])/255)
# plt.ylim(0, 25)
plt.xlim(0, 0.02)
plt.xticks(np.arange(0, 0.021, 0.02))
ax44 = fig.add_subplot(gs[3, 15:20])
plt.hist(gamma, bins=32, color=(0.4, 0.4, 0.4))
plt.title(r'Histogram $\gamma$')
plt.xticks(np.arange(0, 5))
plt.yticks(np.arange(0, 36, 5))
plt.ylim(0, 37)
plt.xlim(0, 4)

plt.subplots_adjust(left=0.05, right=0.97, top=0.93, bottom=0.08, hspace=0.3, wspace=100)
ax41.set_position(matplotlib.transforms.Bbox(ax41.get_position() + np.array([[0, -0.045], [0, -0.045]])))
ax42.set_position(matplotlib.transforms.Bbox(ax42.get_position() + np.array([[0, -0.045], [0, -0.045]])))
ax43.set_position(matplotlib.transforms.Bbox(ax43.get_position() + np.array([[0, -0.045], [0, -0.045]])))
ax44.set_position(matplotlib.transforms.Bbox(ax44.get_position() + np.array([[0, -0.045], [0, -0.045]])))
set_fontsize(label=11, tick=9)
savefig(f'../figures/model/compare_spec_unspec_ext.png', pad_inches=0.01)
plt.show()