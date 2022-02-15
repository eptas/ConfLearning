import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


def savefig(path, relative_to_home=False, bbox_inches='tight', pad_inches=0, dpi=300, **kwargs):
    mpl.rcParams['svg.fonttype'] = 'none'
    if relative_to_home:
        path = os.path.join(os.path.expanduser("~"), path)
    plt.savefig(path, bbox_inches=bbox_inches, pad_inches=pad_inches, dpi=dpi, **kwargs)


def set_fontsize(label=None, xlabel=None, ylabel=None,
        tick=None, xtick=None, ytick=None,
        title=None,
        set=None
        ):

    fig = plt.gcf()

    if set == 'default':
        label = 14
        tick = 12
        title = 16

    for ax in fig.axes:
        if xlabel is not None:
            ax.xaxis.label.set_size(xlabel)
        elif label is not None:
            ax.xaxis.label.set_size(label)
        if ylabel is not None:
            ax.yaxis.label.set_size(ylabel)
        elif label is not None:
            ax.yaxis.label.set_size(label)

        if xtick is not None:
            for ticklabel in (ax.get_xticklabels()):
                ticklabel.set_fontsize(xtick)
        elif tick is not None:
            for ticklabel in (ax.get_xticklabels()):
                ticklabel.set_fontsize(tick)
        if ytick is not None:
            for ticklabel in (ax.get_yticklabels()):
                ticklabel.set_fontsize(ytick)
        elif tick is not None:
            for ticklabel in (ax.get_yticklabels()):
                ticklabel.set_fontsize(tick)

        if title is not None:
            ax.title.set_fontsize(title)