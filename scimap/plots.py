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

import numpy as np
from matplotlib import pyplot, cm
from matplotlib.ticker import ScalarFormatter

from .utilities import q_to_twotheta
# from .xrd.utilities import q_to_twotheta


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


def draw_colorbar(ax, cmap, norm, energies, orientation="vertical",
                  *args, **kwargs):  # pragma: no cover
    """Draw a colorbar on the side of a mapping axes to show the range of
    colors used. Returns the newly created colorbar object.
    Arguments
    ---------
    - ax : Matplotlib axes object against which to plot.
    - cmap : String or mpl Colormap instance indicating which colormap
      to use.
    - norm : mpl Normalize object that describes the range of values to
      use.
    - energies : Iterable of values to put as the tick marks on the
      colorbar.
    """
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.arange(0, 3))
    # Add the colorbar to the axes
    cbar = pyplot.colorbar(mappable,
                           ax=ax,
                           ticks=energies,
                           spacing="proportional",
                           orientation=orientation,
                           *args, **kwargs)
    # Annotate the colorbar
    cbar.ax.set_title('eV')
    # Make sure the ticks don't use scientific notation
    cbar.formatter.set_useOffset(False)
    cbar.update_ticks()
    return cbar


def draw_histogram_colorbar(ax, *args, **kwargs):  # pragma: no cover
    """Similar to `draw_colorbar()` with some special formatting options
    to put it along the X-axis of the axes."""
    cbar = draw_colorbar(ax=ax, pad=0, orientation="horizontal", energies=None, *args, **kwargs)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off',
        labeltop='off')
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
        bottom='on',
        top='on',
        labelbottom="on",
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
               use_twotheta=False, wavelength=None, *args, **kwargs):
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
    use_twotheta : bool, optional
      If true, the results will be plotted with 2θ on the x-axis,
      otherwise scattering length (q) will be used.
    wavelength : float, optional
      The wavelength of radiation to use for converting q to 2θ. Only
      needed if ``use_twotheta`` is truthy.
    *args, **kwargs :
      Passed to matplotlib plot routine.

    """
    if ax is None:
        ax = big_axes()
    scannames = []
    lines = []
    for idx, scan in enumerate(scan_list):
        df = scan.diffractogram.copy()
        df.counts = df.counts + step_size * idx
        if use_twotheta:
            x = q_to_twotheta(np.array(df.index), wavelength=wavelength)
        else:
            x = df.index
        # Do the plotting
        lines.append(ax.plot(x, df.counts, *args, **kwargs)[0])
        # Try and determine a name for this scan
        try:
            scannames.append(names[idx])
        except IndexError:
            scannames.append(getattr(scan, 'name', "Pattern {}".format(idx)))
    ax.legend(reversed(lines), reversed(scannames))
    # Set axes limits
    xMax = max([scan.diffractogram.index.max() for scan in scan_list])
    xMin = min([scan.diffractogram.index.min() for scan in scan_list])
    if use_twotheta:
        xMax, xMin = q_to_twotheta(np.array((xMax, xMin)), wavelength=wavelength)
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
