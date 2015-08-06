# -*- coding: utf-8 -*-

from matplotlib import pyplot

def new_axes(height=5, width=None):
    """Create a new set of matplotlib axes for plotting. Height in inches."""
    # Adjust width to accomodate colorbar
    if width is None:
        width = height/0.8
    fig = pyplot.figure(figsize=(width, height))
    ax = pyplot.gca()
    return ax

def big_axes():
    return new_axes(height=9, width=16)

def dual_axes():
    """Two new axes for mapping, side-by-side."""
    fig, (ax1, ax2) = pyplot.subplots(1, 2)
    fig.set_figwidth(13.8)
    fig.set_figheight(5)
    return (ax1, ax2)
