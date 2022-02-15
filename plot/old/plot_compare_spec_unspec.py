import sys
import os
import matplotlib.pyplot as plt
from plot_model_EC import plot_EC
from plot_model_CPE import plot_CPE
from plot_model_ABSCPE import plot_ABSCPE
from plot_model_value import plot_value

# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))
from plot_util import set_fontsize, savefig  # noqa


fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(3, 11)

ax11 = fig.add_subplot(gs[0, :5])
plot_EC(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoUnspec', title='Expected confidence')
# plt.text(-0.11, 1.04, 'A', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel(None)
plt.xticks(ax11.get_xticks(), [])
plt.text(10, 0.65, 'ConfUnspec', clip_on=False, fontsize=13, fontweight='bold', ha='center', fontstyle='italic')

ax12 = fig.add_subplot(gs[0, 5:10])
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


ax21 = fig.add_subplot(gs[1, :5])
plot_CPE(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoUnspec', title='Confidence prediction error')
# plt.text(-0.11, 1.04, 'C', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel(None)
plt.xticks(ax11.get_xticks(), [])

ax22 = fig.add_subplot(gs[1, 5:10])
plot_CPE(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoSpec', title='Confidence prediction error')
# plt.text(-0.14, 1.04, 'D', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
plt.xlabel(None)
plt.xticks(ax11.get_xticks(), [])

ax31 = fig.add_subplot(gs[2, :5])
plot_ABSCPE(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoUnspec', title='Abs. confidence prediction error')
# plt.text(-0.11, 1.04, 'E', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)

ax32 = fig.add_subplot(gs[2, 5:10])
plot_ABSCPE(legend_value=False, legend_phase1=False, ylabel_as_title=True, reload=False, winning_model='MonoSpec', title='Abs. confidence prediction error')
# plt.text(-0.14, 1.04, 'F', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)

plt.subplots_adjust(left=0.05, right=0.97, top=0.91, bottom=0.07, hspace=0.3, wspace=3)
set_fontsize(label=11, tick=9)
savefig(f'../figures/model/compare_spec_unspec.png', pad_inches=0.01)
plt.show()