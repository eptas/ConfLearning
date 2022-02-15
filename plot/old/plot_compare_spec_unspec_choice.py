import sys
import os
import matplotlib.pyplot as plt
from plot_model_conf import plot_MC
from plot_model_value import plot_value
from plot_model_behavconf import plot_BC

# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))
from plot_util import set_fontsize, savefig  # noqa


fig = plt.figure(figsize=(7, 8))
gs = fig.add_gridspec(4, 11)

ax11 = fig.add_subplot(gs[0, :5])
handles_phase1, labels_phase1, handles_value, labels_value = \
plot_BC(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoUnspec', title='Behavior')
# plt.text(-0.11, 1.04, 'A', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel(None)
plt.ylabel('Confidence')
plt.xticks(ax11.get_xticks(), [])
plt.text(8, 1.3, 'Confidence', clip_on=False, fontsize=13, fontweight='bold', ha='center')

# Plot legends
leg = plt.legend(handles_phase1, labels_phase1, loc='upper left', bbox_to_anchor=(1.1, 1.1), title='No. trials\nin Phase 1:', fontsize=9, title_fontsize=9.5, labelspacing=0.3, handlelength=4, frameon=False)
leg._legend_box.align = 'left'
ax11.add_artist(leg)
leg2 = plt.legend(handles_value, labels_value, loc='upper left', bbox_to_anchor=(1.52, 1), fontsize=9, labelspacing=0.1, handletextpad=2.5, framealpha=1, title='Stimulus:', title_fontsize=9.5)
plt.gca().add_artist(leg2)
plt.arrow(1.66, 0.42, 0, 0.4, color=(0.3, 0.3, 0.3), head_length=0.02, head_width=0.015, length_includes_head=True, clip_on=False, transform=plt.gca().transAxes, zorder=10, lw=0.75)
plt.text(1.72, 0.52, 'Value\nlevel', color=(0.3, 0.3, 0.3), ha='center', rotation=90, transform=plt.gca().transAxes, fontsize=9, zorder=10, linespacing=0.87)


ax21 = fig.add_subplot(gs[1, :5])
plot_MC(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoUnspec', title='ConfUnspec', title_italic=True)
# plt.text(-0.11, 1.04, 'A', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel(None)
plt.xticks(ax11.get_xticks(), [])
plt.ylabel('Confidence')

ax22 = fig.add_subplot(gs[1, 5:10])
plot_value(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoUnspec', title='ConfUnspec', title_italic=True)
# plt.text(-0.11, 1.04, 'D', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel(None)
plt.xticks(ax11.get_xticks(), [])
plt.text(8, 37, 'Expected value', clip_on=False, fontsize=13, fontweight='bold', ha='center')
plt.ylabel('Value')
ax22.yaxis.set_label_coords(-0.095, 0.5)

ax31 = fig.add_subplot(gs[2, :5])
plot_MC(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoSpec', title='ConfSpec', title_italic=True)
# plt.text(-0.14, 1.04, 'B', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel(None)
plt.xticks(ax11.get_xticks(), [])
plt.ylabel('Confidence')

ax32 = fig.add_subplot(gs[2, 5:10])
plot_value(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoSpec', title='ConfSpec', title_italic=True)
# plt.text(-0.14, 1.04, 'E', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel(None)
plt.xticks(ax11.get_xticks(), [])
plt.ylabel('Value')
ax32.yaxis.set_label_coords(-0.095, 0.5)

ax41 = fig.add_subplot(gs[3, :5])
plot_MC(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='Mono_choice', title='Choice', title_italic=True)
# plt.text(-0.14, 1.04, 'C', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.ylabel('Confidence')

ax42 = fig.add_subplot(gs[3, 5:10])
plot_value(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='Mono_choice', title='Choice', title_italic=True)
# plt.text(-0.14, 1.04, 'F', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.ylabel('Value')
ax42.yaxis.set_label_coords(-0.095, 0.5)

plt.subplots_adjust(left=0.08, right=1.08, top=0.92, bottom=0.06, hspace=0.275, wspace=4)
set_fontsize(label=11, tick=9)
savefig(f'../figures/model/compare_unspec_spec_choice.png', pad_inches=0.01)
plt.show()