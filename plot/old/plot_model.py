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


fig = plt.figure(figsize=(10, 5.5))
gs = fig.add_gridspec(2, 16)

ax1 = fig.add_subplot(gs[0, :5])
plot_value(legend_value=False, legend_phase1=False, ylabel_as_title=True)
plt.text(-0.11, 1.04, 'A', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)

ax2 = fig.add_subplot(gs[0, 5:10])
plot_EC(legend_value=False, legend_phase1=False, ylabel_as_title=True)
plt.text(-0.14, 1.04, 'B', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)

ax3 = fig.add_subplot(gs[0, 10:15])
handles_phase1, labels_phase1, handles_value, labels_value = plot_CPE(legend_value=False, legend_phase1=False,
                                                                      ylabel_as_title=True, markersize_value=6)
plt.text(-0.14, 1.04, 'C', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)

# Plot legends
leg = plt.legend(handles_phase1, labels_phase1, loc='upper left', bbox_to_anchor=(1, 1.08), title='No. trials\nin Phase 1:', fontsize=9, title_fontsize=9.5, labelspacing=0.3, handlelength=4, frameon=False)
leg._legend_box.align = 'left'
ax3.add_artist(leg)
leg2 = plt.legend(handles_value, labels_value, loc='upper left', bbox_to_anchor=(1.02, 0.5), fontsize=9, labelspacing=0.1, handletextpad=2.5, framealpha=1, title='Stimulus:', title_fontsize=9.5)
plt.gca().add_artist(leg2)
plt.arrow(1.17, 0.02, 0, 0.335, color=(0.3, 0.3, 0.3), head_length=0.02, head_width=0.015, length_includes_head=True, clip_on=False, transform=plt.gca().transAxes, zorder=10, lw=0.75)
plt.text(1.22, 0.03, 'True value', color=(0.3, 0.3, 0.3), ha='center', rotation=90, transform=plt.gca().transAxes, fontsize=9, zorder=10)


ax4 = fig.add_subplot(gs[1, :4])
plot_histo('alpha')
plt.text(-0.155, 1.04, 'D', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)

ax5 = fig.add_subplot(gs[1, 4:8])
plot_histo('beta')
plt.text(-0.14, 1.04, 'E', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)

ax6 = fig.add_subplot(gs[1, 8:12])
plot_histo('gamma')
plt.text(-0.14, 1.04, 'F', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)

ax7 = fig.add_subplot(gs[1, 12:16])
plot_histo('alpha_c')
plt.text(-0.14, 1.04, 'G', transform=ax7.transAxes, color=(0, 0, 0), fontsize=17)

plt.subplots_adjust(left=0.03, right=0.973, top=0.93, bottom=0.05, hspace=0.5, wspace=3)
set_fontsize(label=11, tick=9)
savefig(f'../figures/model/model.png', pad_inches=0.01)
plt.show()