import pandas as pd
import os
from pathlib import Path

cwd = Path.cwd()
path_data = os.path.join(cwd, '../data/')

models = ['Static', 'Deval', 'Choice', 'ConfSpec', 'ConfUnspec', 'Perseveration']
model_files = ['Static_simchoice', 'Deval_simchoice', 'Mono_choice_simchoice', 'MonoSpec_simchoice', 'MonoUnspec_simchoice', 'Perservation_simchoice']

for i, model in enumerate(models):
    fit = pd.read_pickle(os.path.join(path_data, f"../results/fittingData/fittingData_{model_files[i]}.pkl"))
    fit = fit[~fit.index.isin([25, 30])]
    print('\n', model, '\n')
    df = pd.concat((pd.DataFrame(fit.mean()), pd.DataFrame(fit.sem()), pd.DataFrame(fit.min()), pd.DataFrame(fit.max())), axis=1)
    df.columns = ['Mean', 'SEM', 'Min', 'Max']
    with pd.option_context('display.float_format', '{:0.2f}'.format):
        print(df)