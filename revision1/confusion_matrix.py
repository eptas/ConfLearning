import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm

from sklearn.metrics import ConfusionMatrixDisplay as CMD
from pathlib import Path
import matplotlib.transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import sem

warnings.filterwarnings("ignore")

cwd = Path.cwd()
path_data = os.path.join(cwd, '../data/')

model_names = ['Static', 'Deval', 'Choice', 'ConfSpec', 'ConfUnspec', 'Perservation']
modellist = ['Static', 'Deval', 'Mono_choice', 'MonoSpec', 'MonoUnspec', 'Perservation']
n_models = len(model_names)
fitlist = ['NEGLL', 'AIC', 'BIC']

data = pd.read_pickle('fittingData_' + model_names[(n_models - 1)] + '.pkl')

# plt.savefig("confusion_NEGLL")
# NEGLL = data.filter(regex='NEGLL_').mean(axis=0)
# conf_negll = np.array([NEGLL[0:5].values, NEGLL[5:10].values, NEGLL[10:15].values, NEGLL[15:20].values, NEGLL[20:25].values])
# con_negll = CMD(conf_negll)
# con_negll.plot(values_format='.3f')
# plt.xticks(range(5), model_names)
# plt.yticks(range(5), model_names[::-1])
# plt.tight_layout()
# plt.savefig("confusion_NEGLL")

AIC = data.filter(regex='AIC_').mean(axis=0)
conf_aic = np.array([AIC[0:6].values, AIC[6:12].values, AIC[12:18].values, AIC[18:24].values, AIC[24:30].values, AIC[30:36].values]).T
con_aic = CMD(conf_aic)
aic_plot = con_aic.plot(values_format='.1f', xticks_rotation=345, colorbar=False)
plt.gcf().set_size_inches((1.15*4.5, 1.15*3.9))
plt.xticks(range(n_models), [m.replace('Perservation', 'Perseveration') for m in model_names])
plt.yticks(range(n_models), [m.replace('Perservation', 'Perseveration') for m in model_names])
offset = matplotlib.transforms.ScaledTranslation(-7/72, 2/72, plt.gcf().dpi_scale_trans)
for tick in aic_plot.ax_.xaxis.get_majorticklabels():
    tick.set_horizontalalignment("left")
    tick.set_fontstyle("italic")
    tick.set_fontsize(11)
    tick.set_transform(tick.get_transform() + offset)
for tick in aic_plot.ax_.yaxis.get_majorticklabels():
    tick.set_fontstyle("italic")
    tick.set_fontsize(11)
plt.xlabel('Fitted model', fontsize=12)
plt.ylabel('Generative model', fontsize=12)
aic_plot.ax_.yaxis.set_label_coords(-0.35, 0.5)
cb = plt.colorbar(aic_plot.im_, make_axes_locatable(aic_plot.ax_).append_axes("right", size="5%", pad=0.05))
cbl = cb.set_label('AIC', fontsize=12, labelpad=3)

plt.tight_layout()
plt.savefig("confusion_AIC", bbox_inches='tight', pad_inches=0, dpi=300)

# BIC = data.filter(regex='BIC_').mean(axis=0)
# conf_aic = np.array([BIC[0:5].values, BIC[5:10].values, BIC[10:15].values, BIC[15:20].values, BIC[20:25].values])
# con_aic = CMD(conf_aic)
# con_aic.plot(values_format='.3f')
# plt.xticks(range(5), model_names)
# plt.yticks(range(5), model_names[::-1])
# plt.tight_layout()
# plt.savefig("confusion_BIC")
