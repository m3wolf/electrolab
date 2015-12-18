# -*- coding: utf-8 -*-

from matplotlib import pyplot
from matplotlib.ticker import ScalarFormatter

class ElectronVoltFormatter(ScalarFormatter):
    """Matplotlib formatter for showing energies as electon-volts."""
    def __call__(self, *args, **kwargs):
        formatted_value = super().__call__(*args, **kwargs)
        formatted_value = "{value} eV".format(value=formatted_value)
        return formatted_value


class DegreeFormatter(ScalarFormatter):
    """Matplotlib formatter for showing angles with the degree symbol."""
    def __call__(self, *args, **kwargs):
        formatted_value = super().__call__(*args, **kwargs)
        formatted_value = "{value}°".format(value=formatted_value)
        return formatted_value


def remove_extra_spines(ax):
    """Removes the right and top borders from the axes."""
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    return ax

def new_axes(height=5, width=None):
    """Create a new set of matplotlib axes for plotting. Height in inches."""
    # Adjust width to accomodate colorbar
    if width is None:
        width = height/0.8
    fig = pyplot.figure(figsize=(width, height))
    # Set background to be transparent
    fig.patch.set_alpha(0)
    # Create axes
    ax = pyplot.gca()
    # Remove borders
    remove_extra_spines(ax)
    return ax

def big_axes():
    return new_axes(height=9, width=16)

def xrd_axes():
    return new_axes(width=8)

def dual_axes(orientation='horizontal'):
    """Two new axes for mapping, side-by-side."""
    if orientation == 'vertical':
        fig, (ax1, ax2) = pyplot.subplots(2, 1)
        fig.set_figwidth(6.9)
        fig.set_figheight(13.8)
    else:
        fig, (ax1, ax2) = pyplot.subplots(1, 2)
        fig.set_figwidth(13.8)
        fig.set_figheight(5)
    # Remove redundant borders
    remove_extra_spines(ax1)
    remove_extra_spines(ax2)
    return (ax1, ax2)

def plot_scans(scan_list, step_size=0, ax=None):
    """
    Plot a series of XRDScans as a waterfall. step_size controls how
    far apart the waterfall stacking is. Optional keyword arg 'ax' plots
    on a specific Axes.
    """
    if ax is None:
        ax = big_axes()
    scannames = []
    lines = []
    for idx, scan in enumerate(scan_list):
        df = scan.diffractogram.copy()
        df.counts = df.counts + step_size * idx
        lines.append(ax.plot(df.index, df.counts)[0])
        scannames.append(scan.name)
    ax.legend(reversed(lines), reversed(scannames))
    # Set axes limits
    df = scan_list[0].diffractogram
    xMax = max([scan.diffractogram.index.max() for scan in scan_list])
    xMin = min([scan.diffractogram.index.min() for scan in scan_list])
    ax.set_xlim(left=xMin, right=xMax)
    # Decorate
    ax.set_xlabel(r'$2\theta$')
    ax.set_ylabel('counts')
    return ax

def plot_txm_intermediates(images):
    for key in images.keys():
        ax1, ax2 = dual_axes()
        ax1.imshow(images[key], cmap='gray')
        ax1.set_title(key)
        ax2.hist(images[key].flat, bins=100)
