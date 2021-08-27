import pandas as pd

from ConfLearning.util.model_to_latex import latex_to_png
import os
from pathlib import Path
import numpy as np

# file, DV, pngname = 'model_13_corr.csv', 'correct', 'performance_effect'
# file, DV, pngname = 'model_2_corr.csv', 'correct', 'performance_effect_phase2'
file, DV, pngname = 'model_2_consist.csv', 'consistent', 'consistency_effect_phase2'
df = pd.read_csv(file)

header = '\\begin{table}\n\\begin{center}\n\\begin{tabular}{lrrrrrr}\n\\hline\n'
footer = '\\hline\n\\end{tabular}\n\\end{center}\n\\end{table}'
firstline = '\\textit{DV: ' + DV + '} & Coef. & Std.Err. & z & P$> |$z$|$ & [0.025 & 0.975] \\\\\n\\hline\n'
# IVs = ['Intercept', 'block\\_difficulty', 'block\\_value\\_level']
# coef, se, z, p, ci = np.ones(3), np.ones(3), np.ones(3), np.ones(3), np.ones(3)
body = ''
for i in range(len(df)):
    name, coef, se, z, p, ci1, ci2 = df.iloc[i].values
    name = name.replace('(Intercept)', 'Intercept').replace('_', '\\_')
    body += f'{name} & {coef:.3f} & {se:.3f} & {z:.3f} & {p:.3f} & {ci1:.3f} & {ci2:.3f}' + ' \\\\\n'

re = 'subject Var & 0.001 & 0.001 & & & & \\\\\n' \
     'block Var & 0.013 & 0.003 & & & & \\\\\n'


content = header + firstline + body + footer

latex_to_png(content, outpath=os.path.join(os.getcwd(), 'regtables', f"{pngname}.png"),
             title=None, DV='correct')