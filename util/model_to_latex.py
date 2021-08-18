import re
from statsmodels.iolib.summary2 import _simple_tables
from matplotlib import rcParams
import os
from pathlib import Path

def as_latex(model, title=None, DV=None, print_meta=True, single_table=False):
    '''Generate LaTeX Summary Table
    '''
    tables = model.tables
    settings = model.settings

    if title is not None:
        title = r'\caption*{' + title + '}'

    simple_tables = _simple_tables(tables, settings)
    if not print_meta:
        simple_tables = simple_tables[1:]
    tab = [x.as_latex_tabular() for x in simple_tables]
    tab = '\n\\hline\n'.join(tab)

    to_replace = ('\\\\hline\\n\\\\hline\\n\\\\'
                  'end{tabular}\\n\\\\begin{tabular}{.*}\\n')

    if model._merge_latex:
        # create single tabular object for summary_col
        tab = re.sub(to_replace, r'\\midrule\n', tab)

    if title is not None:
        out = '\\begin{table}', title, tab, '\\end{table}'
    else:
        out = '\\begin{table}', tab, '\\end{table}'
    out = '\n'.join(out)
    if single_table:
        out = out.replace('\\end{tabular}\n\\begin{tabular}{lrrrrrr}\n', '')
    if DV is not None:
        if '& Coef.' in out:
            out = out.replace('& Coef.', '\\textit{DV: ' + DV + '} & Coef.')
        else:
            out = out.replace('&  Coef.', '\\textit{DV: ' + DV + '} & Coef.')
    return out


def latex_to_png(model, outpath='out.png', title=None, DV=None):
    rcParams['text.usetex'] = True

    beginningtex = r"""\documentclass[preview]{standalone}\thispagestyle{empty}\usepackage{booktabs}\usepackage[font=bf,aboveskip=0pt]{caption}\begin{document}"""
    endtex = r"\end{document}"
    content = as_latex(model.summary(), print_meta=False, single_table=True, title=title, DV=DV)
    latex = beginningtex + content + endtex

    f = open('latex/document.tex', 'w')
    f.write(latex)
    f.close()
    os.system('pdflatex -output-directory latex latex/document.tex  > /dev/null 2>&1 && pdfcrop latex/document.pdf latex/document.pdf > /dev/null 2>&1 && pdftoppm -r 300 latex/document.pdf|pnmtopng > latex/document.png')
    os.rename('latex/document.png', outpath)

    rcParams['text.usetex'] = False