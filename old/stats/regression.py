import pandas as pd
import statsmodels
import statsmodels.api as sm
import warnings
from scipy.stats import pearsonr, linregress, spearmanr, theilslopes, zscore
from statsmodels.base.model import Model
from statsmodels.formula.formulatools import handle_formula_data
import numpy as np
import re
from IPython.display import display
from mg.plot.notebook import t as nbprint
from functools import partial
import pingouin as pg
from collections import namedtuple
from sklearn.linear_model import RANSACRegressor

# warnings.filterwarnings('error')

pd.options.mode.chained_assignment = None

def linear_regression(data, patsy_string, standardize_vars=True, print_summary=True,
                      model_blocks=None, ignore_warnings=True,
                      print_patsy=True, print_corr_table=True, print_data=False, print_extra=True,
                      vars_corr=None,
                      reml=True,
                      notebook_print=False,
                      silent=False, print_short=False, ols=False,
                      standardize_vars_excl=(),
                      groupname='subject', return_model=True, return_data=False):

    if model_blocks is None:
        model_blocks = 'block' in data

    DV = patsy_string.split('~')[0].strip()
    IVs = list(dict.fromkeys([iv.split('(')[1].split(')')[0] if '(' in iv else iv.strip() for iv in patsy_string.split('~')[1].replace('*', '+').replace(':', '+').replace('-', '+').split('+')]))

    allVars = [DV] + IVs + [groupname]
    if model_blocks:
        allVars += ['block']

    dtypes = dict()
    for var in allVars:
        if isinstance(data[var].dtype, (pd.Int64Dtype, object)):
            dtypes.update({var: float})
        if data[var].dtype == np.dtype('bool'):
            dtypes.update({var: float})
    data = data.astype(dtypes).reset_index()

    if standardize_vars:
        for var in allVars:
            if not var in ([groupname] if 'block' in patsy_string else [groupname, 'block']) and not var in standardize_vars_excl:
                data[var] = ((data[var] - data[var].mean()) / data[var].std()).values

    if return_data:
        return data

    print_ = partial(nbprint, family='monospace') if notebook_print else print

    if not silent and print_data:
        print_(data[allVars])
        for v in allVars:
            print_({v: data[v].dtype})

    if not silent and print_corr_table:
        vars = [DV] + IVs if vars_corr is None else vars_corr
        corrtab = data[vars].corr()
        pcorrtab = data[vars].corr(method=lambda x, y: pearsonr(x, y)[1])
        corrtab_sorted = dict(sorted({k:v for el in [{f'{c1} x {c2}': corrtab.loc[c1, c2] for c2 in corrtab.columns if list(corrtab.columns).index(c2) > list(corrtab.columns).index(c1)} for c1 in corrtab.columns] for k,v in el.items()}.items(), key=lambda x: x[1]))
        pcorrtab_sorted = dict(sorted({k:v for el in [{f'{c1} x {c2}': pcorrtab.loc[c1, c2] for c2 in pcorrtab.columns if list(pcorrtab.columns).index(c2) > list(pcorrtab.columns).index(c1)} for c1 in pcorrtab.columns] for k,v in el.items()}.items(), key=lambda x: x[1]))
        for k, v in corrtab_sorted.items():
            print_(f'{k}: {v:.3f} (p={pcorrtab_sorted[k]:.4f})')

    with warnings.catch_warnings():
        if ignore_warnings:
            warnings.simplefilter('ignore')
        if model_blocks:
            model = sm.MixedLM.from_formula(patsy_string, groups=groupname, re_formula='1', vc_formula={'block': '0+C(block)'}, data=data).fit(reml=reml)
        elif ols:
            model = sm.OLS.from_formula(patsy_string, data=data).fit(reml=reml)
        else:
            model = sm.MixedLM.from_formula(patsy_string, groups=groupname, data=data).fit(reml=reml)
        if not silent:
            if print_patsy:
                print_(patsy_string + '\n')
            if print_summary:
                if print_short:
                    a = str(model.summary())
                    if ols:
                        pos = np.array([v.start() for v in re.finditer('=', a)])
                        pos_parts = np.where(np.diff(pos) > 1)[0]
                        print_('='*80 + '\n' + a[pos[pos_parts[1]]+2:pos[pos_parts[2]]+1] + '==')
                    else:
                        pos2 = np.array([v.start() for v in re.finditer('--', a)])
                        pos_parts2 = np.where(np.diff(pos2) > 2)[0]
                        print_(a[pos2[pos_parts2[0]]+2:])
                else:
                    print_(model.summary())
                    if print_extra and (model.aic is not None):
                        print_(f'AIC: {model.aic} BIC: {model.bic}')

    predictions = model.predict(data)
    r, p = pearsonr(predictions, data[DV])
    if not silent and not print_short and print_extra:
        print_(f'Pearson r = {r} (p = {p})')

        if len(data[~data[DV].isna()][DV].unique()) == 2:
            print_(f'Accuracy: {np.mean((data[~data[DV].isna()][DV].values > 0) == (predictions > 0))}')


    if return_model:
        return model
