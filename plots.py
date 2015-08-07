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

def plot_scans(scan_list, step_size=1, ax=None):
    """
    Plot a series of XRDScans as a waterfall. step_size controls how
    far apart the waterfall stacking is. Optional keyword arg 'ax' plots
    on a specific Axes.
    """
    if ax is None:
        ax = plots.big_axes()
    scannames = []
    lines = []
    for idx, scan in enumerate(scan_list):
        df = scan.diffractogram.copy()
        df.counts = df.counts + step_size * idx
        lines.append(ax.plot(df.index, df.counts)[0])
        scannames.append(scan.name)
    ax.legend(reversed(lines), reversed(scannames))
    # Set axes limits
    xMax = scan_list[0].diffractogram.index.max()
    xMax = max([scan.diffractogram.index.max() for scan in scan_list])
    ax.set_xlim(right=xMax)
    # Decorate
    ax.set_xlabel(r'$2\theta$')
    ax.set_ylabel('counts')
