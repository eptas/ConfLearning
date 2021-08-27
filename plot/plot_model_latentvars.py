import sys
import os
import matplotlib.pyplot as plt
from plot_model_value import plot_value
from plot_model_EC import plot_EC
from plot_model_CPE import plot_CPE
from histo_params_winning import plot_histo

# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))
from plot_util import set_fontsize, savefig  # noqa


fig = plt.figure(figsize=(10, 3))
gs = fig.add_gridspec(1, 16)

ax1 = fig.add_subplot(gs[0, :5])
plot_value(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False)
plt.text(-0.11, 1.04, 'A', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)

ax2 = fig.add_subplot(gs[0, 5:10])
plot_EC(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False)
plt.text(-0.14, 1.04, 'B', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)

ax3 = fig.add_subplot(gs[0, 10:15])
handles_phase1, labels_phase1, handles_value, labels_value = \
    plot_CPE(legend_value=False, legend_phase1=False, ylabel_as_title=True, markersize_value=6, reload=False)
plt.text(-0.14, 1.04, 'C', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)

# Plot legends
leg = plt.legend(handles_phase1, labels_phase1, loc='upper left', bbox_to_anchor=(1, 1.08), title='No. trials\nin Phase 1:', fontsize=9, title_fontsize=9.5, labelspacing=0.3, handlelength=4, frameon=False)
leg._legend_box.align = 'left'
ax3.add_artist(leg)
leg2 = plt.legend(handles_value, labels_value, loc='upper left', bbox_to_anchor=(1.02, 0.5), fontsize=9, labelspacing=0.1, handletextpad=2.5, framealpha=1, title='Stimulus:', title_fontsize=9.5)
plt.gca().add_artist(leg2)
plt.arrow(1.17, 0.142, 0, 0.228, color=(0.3, 0.3, 0.3), head_length=0.02, head_width=0.015, length_includes_head=True, clip_on=False, transform=plt.gca().transAxes, zorder=10, lw=0.75)
plt.text(1.24, 0.19, 'Value\nlevel', color=(0.3, 0.3, 0.3), ha='center', rotation=90, transform=plt.gca().transAxes, fontsize=9, zorder=10, linespacing=0.87)

plt.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.15, hspace=0.5, wspace=3)
set_fontsize(label=11, tick=9)
savefig(f'../figures/model/model_latentvars.png', pad_inches=0.01)
plt.show()