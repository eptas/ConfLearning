import sys
import os
import matplotlib.pyplot as plt
from plot_corr_alpha_gamma_winning import plot_corr_alpha_gamma
from histo_params_winning import plot_histo
from matplotlib.transforms import Bbox
import numpy as np

# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))
from plot_util import set_fontsize, savefig  # noqa


fig = plt.figure(figsize=(8, 5))
gs = fig.add_gridspec(2, 3)

ax4 = fig.add_subplot(gs[0, 0])
plot_histo('alpha')
plt.text(-0.17, 1.04, 'A', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)

ax5 = fig.add_subplot(gs[0, 1])
plot_histo('beta')
plt.text(-0.16, 1.04, 'B', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)

ax6 = fig.add_subplot(gs[0, 2])
plot_histo('gamma')
plt.text(-0.16, 1.04, 'C', transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)

ax7 = fig.add_subplot(gs[1, 0])
plot_histo('alpha_c')
plt.text(-0.14, 1.04, 'D', transform=ax7.transAxes, color=(0, 0, 0), fontsize=17)

ax7 = fig.add_subplot(gs[1, 1:3])
plot_corr_alpha_gamma()
plt.text(-0.33, 1.04, 'E', transform=ax7.transAxes, color=(0, 0, 0), fontsize=17)


set_fontsize(label=11, tick=9)
plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.1, hspace=0.4, wspace=0.3)
ax7.set_position(Bbox(ax7.get_position()+ np.array([[0.13, 0], [-0.13, 0]])))
savefig(f'../figures/model/model_params.png', pad_inches=0.01)
plt.show()