# -*- coding: utf-8 -*-
#
# Copyright © 2016 Mark Wolf
#
# This file is part of scimap.
#
# Scimap is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Scimap is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

"""Helper functions for setting up and displaying plots using matplotlib."""

from contextlib import contextmanager
from typing import List, NoReturn

import numpy as np
from matplotlib import pyplot, cm, rcParams, rc_context, style
from matplotlib.ticker import ScalarFormatter

from .utilities import q_to_twotheta, twotheta_to_q


@contextmanager
def latexify(styles: List[str]=[], preamble: List[str]=[]):
    """Set some custom options for saving matplotlib graphics in PGF
    format.
    
    Use this as a context manager, along with additional matplotlib styles:
    
    .. code:: python
        
        with xp.latexify(['beamer']):
            plt.plot(...)
            
    
    This will let you add in LaTeX tools and mpl styles together. By
    default, ``siunitx`` and ``mhchem`` packages are
    included. Additional ``\\usepackage`` statements can be included
    using the ``preamble`` parameter.
    
    Parameters
    ==========
    styles : optional
      Additional matplotlib styles in load in the context.
    preamble : optional
      Additional lines to add to the LaTeX preamble.
    
    """
    # Set default LaTeX PGF style
    pgf_with_latex = {                      # setup matplotlib to use latex for output# {{{
        "pgf.texsystem": "xelatex",        # change this if using xetex or lautex
        # "font.family": "serif",
        "font.serif": [],                   # blank entries should cause plots 
        "font.sans-serif": [],              # to inherit fonts from the document
        "font.monospace": [],
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts 
            r"\usepackage[T1]{fontenc}",        # plots will be generated
            r"\usepackage{fontspec}",
            r"\usepackage[detect-all,locale=DE,per-mode=reciprocal]{siunitx}",
            r"\usepackage[version=4]{mhchem}",
        ] + preamble,
    }
    # Enter the context library
    with rc_context(rc=pgf_with_latex):
        style.use(styles)
        yield


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


def draw_colorbar(ax, cmap, norm, ticks=None, orientation="vertical",
                  *args, **kwargs):  # pragma: no cover
    """Draw a colorbar on the side of a mapping axes to show the range of
    colors used. Returns the newly created colorbar object.
    Arguments
    ---------
    ax : 
      Matplotlib axes object against which to plot.
    cmap : str
      String or mpl Colormap instance indicating which colormap to
      use.
    norm :
      mpl Normalize object that describes the range of values to use.
    ticks : optional
      Iterable of values to put as the tick marks on the colorbar.
    
    """
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.arange(0, 3))
    # Add the colorbar to the axes
    cbar = pyplot.colorbar(mappable,
                           ax=ax,
                           ticks=ticks,
                           spacing="proportional",
                           orientation=orientation,
                           *args, **kwargs)
    # Make sure the ticks don't use scientific notation
    cbar.formatter.set_useOffset(False)
    cbar.update_ticks()
    return cbar


def draw_histogram_colorbar(ax, *args, **kwargs):  # pragma: no cover
    """Similar to `draw_colorbar()` with some special formatting options
    to put it along the X-axis of the axes."""
    cbar = draw_colorbar(ax=ax, pad=0, orientation="horizontal", ticks=None, *args, **kwargs)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        labeltop=False)
    ax.spines['bottom'].set_visible(False)
    cbar.ax.set_xlabel(ax.get_xlabel())
    ax.xaxis.set_visible(False)
    cbar.ax.set_title("")
    cbar.outline.set_visible(False)
    gray = (0.1, 0.1, 0.1)
    cbar.ax.axhline(cbar.ax.get_ylim()[1], linewidth=2, linestyle=":", color=gray)
    cbar.ax.tick_params(
        axis='x',
        which='both',
        bottom=True,
        top=True,
        labelbottom=True,
    )
    return cbar


def remove_extra_spines(ax):
    """Removes the right and top borders from the axes."""
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    return ax


def set_outside_ticks(ax):
    """Convert all the axes so that the ticks are on the outside and don't
    obscure data."""
    ax.get_yaxis().set_tick_params(which='both', direction='out')
    ax.get_xaxis().set_tick_params(which='both', direction='out')
    return ax


def new_axes(height=5, width=None):
    """Create a new set of matplotlib axes for plotting. Height in inches."""
    # Adjust width to accomodate colorbar
    if width is None:
        width = height / 0.8
    fig = pyplot.figure(figsize=(width, height))
    # Set background to be transparent
    fig.patch.set_alpha(0)
    # Create axes
    ax = pyplot.gca()
    # Remove borders
    remove_extra_spines(ax)
    return ax


def new_image_axes(height=5, width=5):
    """Square axes with ticks on the outside."""
    ax = new_axes(height, width)
    return set_outside_ticks(ax)


def big_axes():
    """Return a new Axes object, but larger than the default."""
    return new_axes(height=9, width=16)


def xrd_axes():
    """Return a new Axes object, with a size appropriate for display x-ray
    diffraction data."""
    return new_axes(width=8)


def dual_axes(fig=None, orientation='horizontal'):
    """Two new axes for mapping, side-by-side."""
    if fig is None:
        fig = pyplot.figure()
    if orientation == 'vertical':
        fig, (ax1, ax2) = fig.subplots(2, 1)
        fig.set_figwidth(6.9)
        fig.set_figheight(13.8)
    else:
        fig, (ax1, ax2) = pyplot.subplots(1, 2)
        fig.set_figwidth(13.8)
        fig.set_figheight(5)
    # Remove redundant borders
    remove_extra_spines(ax1)
    remove_extra_spines(ax2)
    # Set background to be transparent
    fig.patch.set_alpha(0)
    return (ax1, ax2)


def plot_scans(scan_list, step_size=0, ax=None, names=[],
               use_twotheta=False, wavelength=None, colors=(),
               normalize=False, *args, **kwargs):
    """Plot a series of XRDScans as a waterfall.
    
    Parameters
    ==========
    scan_list : list
      The ``XRDScan`` objects to be plotted.
    step_size : float, int, optional
      How much space to put between successive scans.
    ax : Axes, optional
      The matplotlib Axes object to receive the plots. If omitted, a
      new Axes will be created.
    names : list(str), optional
      Legend entries to use for these scans. If omitted, the names
      will be retried from the XRDScan objects.
    colors : tuple, optional
      Iterable of maptlotlib colors to use for plotting, with the same
      number of entries as ``scan_list``.
    use_twotheta : bool, optional
      If true, the results will be plotted with 2θ on the x-axis,
      otherwise scattering length (q) will be used.
    wavelength : float, optional
      The wavelength of radiation to use for converting q to 2θ. Only
      needed if ``use_twotheta`` is truthy.
    normalize : str, optional
      If 'area' or 'height', normalize each scan by the area under the
      scan.
    *args, **kwargs :
      Passed to matplotlib plot routine.
    
    """
    if ax is None:
        ax = big_axes()
    scannames = []
    lines = []
    for idx, scan in enumerate(scan_list):
        df = scan.diffractogram.copy()
        if not use_twotheta:
            x = twotheta_to_q(np.array(df.index), wavelength=wavelength)
        else:
            x = df.index
        # Prepare the y data
        y = df.counts
        if normalize == 'area':
            y = (y - np.min(y)) / np.trapz(y, x=x)
        elif normalize == 'height':
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
        y += step_size * idx
        # Check if a color is specified
        try:
            color = colors[idx]
        except (IndexError, TypeError):
            color = None
        # Plot the data
        zorder = len(scan_list) - idx
        lines.append(ax.plot(x, y, color=color, zorder=zorder, *args, **kwargs)[0])
        # Try and determine a name for this scan
        try:
            scannames.append(names[idx])
        except IndexError:
            scannames.append(getattr(scan, 'name', "Pattern {}".format(idx)))
    ax.legend(reversed(lines), reversed(scannames))
    # Set axes limits
    xMax = max([scan.diffractogram.index.max() for scan in scan_list])
    xMin = min([scan.diffractogram.index.min() for scan in scan_list])
    if not use_twotheta:
        xMax, xMin = twotheta_to_q(np.array((xMax, xMin)), wavelength=wavelength)
    ax.set_xlim(left=xMin, right=xMax)
    # Decorate
    if use_twotheta:
        ax.set_xlabel(r'$2\theta$')
        ax.xaxis.set_major_formatter(DegreeFormatter())
    else:
        ax.set_xlabel(r'q /$A^{-}$')
        ax.set_xlim(left=xMin, right=xMax)
        # ax.set_xlabel('q /⁻')
    ax.set_ylabel('Counts')
    return ax


def plot_txm_intermediates(images):
    """Accept a dictionary of images and plots them each on its own
    axes. This is a complement to routines that operate on a
    microscopy frame and optionally return all the intermediate
    calculated frames.
    """
    for key in images.keys():
        ax1, ax2 = dual_axes()
        ax1.imshow(images[key], cmap='gray')
        ax1.set_title(key)
        ax2.hist(images[key].flat, bins=100)


def add_twinx(ax, wavelength: float, use_twotheta: bool=True):
    """Add a set of tick marks to the top of axes with complementary values.
    
    If ``use_twotheta`` is True, then the primary x-axis is assumed to
    be in q (scattering length), and 2θ will be plotted on the
    twin-x. Otherwise, the primary axis is assumed to be in 2θ and
    then q will be plotted on the twin-x.
    
    Parameters
    ==========
    ax
      A matplotlib axes to receive the plotting.
    wavelength
      Wavelength of X-ray used for conversion from q to 2θ, in Å.
    use_twotheta
      Whether to use 2θ or q, as described above.
    
    Returns
    =======
    ax2
      The twinx matplotlib.Axes object with the second set of ticks.
    
    """
    q_label = r'q /$A^{-}$'
    two_theta_label = r'$2\theta\ (\lambda={:.4f}\AA)$'
    # Create a new twinx axes to receive new ticks
    ax2 = ax.twiny()
    # Determine new tick labels
    xticks = ax.get_xticks()
    if use_twotheta:
        xticks2 = q_to_twotheta(xticks, wavelength=wavelength)
        xticks2 = ['{:.1f}°'.format(t) for t in xticks2]
        if wavelength is not None:
            xlabel = two_theta_label.format(wavelength)
        else:
            xlabel = two_theta_label.format('??')
    else:
        xticks2 = twotheta_to_q(xticks, wavelength=wavelength)
        xticks2 = [round(t, 2) for t in xticks2]
        xlabel = q_label
    # Set the new tick labels
    ax2.set_xlabel(xlabel)
    ax2.set_xticklabels(xticks2)
    ax2.set_xlim(ax.get_xlim())
    return ax2
