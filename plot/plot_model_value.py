import os
import sys
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sns
from ConfLearning.run_model.run_model_simchoice import run_model, RescorlaConfBaseGen

# This is a trick to import local packages (without Pycharm complaining)
sys.path.append(os.path.dirname(__file__))
from plot_util import set_fontsize, savefig  # noqa

data = pd.read_pickle(os.path.join(Path.cwd(), '../data/', 'data.pkl'))

cwd = Path.cwd()
path_data = os.path.join(cwd, '../results/fittingData')

models = np.arange(1, 10)
nsubjects = 66
nblocks = 11
include = np.setdiff1d(range(nsubjects), [25, 30])

colors = sns.color_palette()


ntrials_phase0 = (9, 12, 15, 18)
ntrials_phase1 = (0, 5, 10, 15)
ntrials_phase2 = (9, 12, 15, 18)
nt_phase1_max = np.max(ntrials_phase1)
nt_phase2_max = np.max(ntrials_phase2)
nt_phase0phase1 = 27

colors = sns.color_palette()
linestyles = [(0, (1, 0.5)), (0, (4, 1)), '-.', '-']

def get_data(winning_model, model_suffix, reload=False):

    m = pd.read_pickle(os.path.join(path_data, f"fittingData_{winning_model}{model_suffix}.pkl"))
    if reload:
        d = data[data.type_choice_obs][['subject', 'block', 'phase', 'trial_phase', 'trial_phase_rev', 'choice', 'b_ntrials_pre', 'b_ntrials_noc']].copy()
        for s in range(nsubjects):
            print(f'Subject {s + 1} / {nsubjects}')
            params = [m.ALPHA[s], m.BETA[s], m.ALPHA_C[s], m.GAMMA[s]]
            val = np.moveaxis(run_model(params, RescorlaConfBaseGen, s, return_cp=False, return_full=True)[0], 2, 3)
            for b in range(nblocks):
                for p in range(3):
                    for c in range(5):
                        d.loc[(d.subject == s) & (d.block == b) & (d.phase == p), f'value{c}'] = val[b, p, c, ~np.isnan(val[b, p, c])]
        d.to_pickle('values.pkl')
    else:
        d = pd.read_pickle('values.pkl')

    return d


def plot_value(legend_phase1=True, legend_value=True, ylabel_as_title=False, markersize_value=10,
               winning_model='MonoUnspec', model_suffix='_simchoice', select_ntrials_phase1=15, reload=False):

    d = get_data(winning_model, model_suffix, reload)
    d = d[(d.b_ntrials_noc == select_ntrials_phase1) & d.subject.isin(include)]

    # dummy plots for legend
    handles_linestyle = [None] * 4
    for i in range(4):
        handles_linestyle[i] = plt.plot([100, 102], [1, 1], lw=2, color=(0.3, 0.3, 0.3), alpha=0.6, ls=linestyles[i])[0]
    handles_color = [None] * 5
    for i in range(5):
        handles_color[i] = plt.plot([100], [1], 's', markersize=markersize_value, mfc=colors[i], alpha=0.6, label='')[0]

    for c in range(5):
        for i, nt in enumerate(ntrials_phase0):
            d0 = d[(d.b_ntrials_pre == nt) & (d.phase == 0)].groupby(['subject', 'trial_phase_rev'])[f'value{c}'].mean()
            m = d0.mean(level='trial_phase_rev').values.astype(float)
            se = d0.sem(level='trial_phase_rev').values.astype(float)
            plt.plot(np.arange(-nt+2, 2), m, lw=2, color=colors[c], alpha=0.6, ls=linestyles[i])
            # plt.fill_between(np.arange(-nt+1, 1), m-se, m+se, lw=0, color=colors[c], alpha=0.4)

        plt.axvspan(1, select_ntrials_phase1, facecolor='0.9', alpha=0.5)
        # plt.axhspan(0, 0.5, facecolor='0.85', alpha=0.5)
        for i, nt in enumerate(ntrials_phase0):
            d1 = d[(d.b_ntrials_pre == nt) & (d.phase == 1)].groupby(['subject', 'trial_phase'])[f'value{c}'].mean()
            m = d1.mean(level='trial_phase').values.astype(float)
            se = d1.sem(level='trial_phase').values.astype(float)
            plt.plot(np.arange(1, select_ntrials_phase1+1), m, lw=2, color=colors[c], alpha=0.6, ls=linestyles[i])
            # plt.fill_between(np.arange(1, nt_phase1_max+1), m-se, m+se, lw=0, color=colors[c], alpha=0.4)
        # d1 = d[(d.phase == 1)].groupby(['subject', 'trial_phase'])[f'value{c}'].mean()
        # m = d1.mean(level='trial_phase').values.astype(float)
        # plt.plot(np.arange(1, nt_phase1_max+1), m, lw=3, color='k', alpha=0.6)

        for i, nt in enumerate(ntrials_phase0):
            d2 = d[(d.b_ntrials_pre == nt) & (d.phase == 2)].groupby(['subject', 'trial_phase'])[f'value{c}'].mean()
            m = d2.mean(level='trial_phase').values.astype(float)
            se = d2.sem(level='trial_phase').values.astype(float)
            plt.plot(np.arange(select_ntrials_phase1, select_ntrials_phase1+nt_phase0phase1-nt), m, lw=2, color=colors[c], alpha=0.6, ls=linestyles[i])
            # plt.fill_between(np.arange(nt_phase1_max+1, nt_phase1_max+nt_phase0phase1-nt+1), m-se, m+se, lw=0, color=colors[c], alpha=0.4)

    if ylabel_as_title:
        plt.title('Model value')
    else:
        plt.ylabel('Model value')
    plt.xticks(np.arange(-20, 40, 5))
    y_text = -1.3
    plt.xlim(-20, 35)
    plt.ylim(-1.8, 28)
    plt.text(-10, y_text, 'Phase 1', ha='center', fontsize=11)
    plt.text(select_ntrials_phase1/2+0.5, y_text, 'Phase 2', ha='center', fontsize=11)
    plt.text(9+select_ntrials_phase1, y_text, 'Phase 3', ha='center', fontsize=11)
    plt.xlabel('Trial')


    handles_phase1, labels_phase1 = handles_linestyle[::-1], ntrials_phase0[::-1]
    handles_value, labels_value = handles_color[::-1], ['' for _ in range(5)]

    if legend_phase1:
        leg = plt.legend(handles_phase1, labels_phase1, loc='upper left', title='No. trials in Phase 1', fontsize=9, title_fontsize=9.5, labelspacing=0.5, handlelength=4, frameon=False)
        leg._legend_box.align = 'left'
        plt.gca().add_artist(leg)
    if legend_value:
        leg2 = plt.legend(handles_value, labels_value, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, labelspacing=0.5, handletextpad=2.5, framealpha=1, title='Stimulus', title_fontsize=9.5)
        plt.gca().add_artist(leg2)
        plt.arrow(1.13, 0.64, 0, 0.263, color=(0.3, 0.3, 0.3), head_length=0.02, head_width=0.015, length_includes_head=True, clip_on=False, transform=plt.gca().transAxes, zorder=10, lw=0.75)
        plt.text(1.16, 0.65, 'True value', color=(0.3, 0.3, 0.3), ha='center', rotation=90, transform=plt.gca().transAxes, fontsize=9, zorder=10)

    return handles_phase1, labels_phase1, handles_value, labels_value

if __name__ == '__main__':


    winning_model = 'MonoUnspec'
    # suffix = ''
    suffix = '_simchoice'
    # suffix = '_cp_simchoice'
    select_ntrials_phase1 = 15

    reload = False

    plt.figure(figsize=(5.5, 4))

    plot_value(winning_model=winning_model, model_suffix=suffix, select_ntrials_phase1=select_ntrials_phase1,
             reload=reload)

    set_fontsize(label=12, tick=10)
    plt.tight_layout()
    plt.subplots_adjust(right=0.84)
    savefig(f'../figures/model/model_value_over_trials_{select_ntrials_phase1}.png', pad_inches=0.01)
    plt.show()