# -*- coding: utf-8 -*-

import os

import pandas as pd
from scipy.interpolate import UnivariateSpline
import scipy
from matplotlib import pyplot


class Reflection():
    """An XRD reflection with a specific hkl value."""
    pass


class Phase():
    """A crystallographic phase that can be found in a Material."""
    pass


class XRDScan():
    """
    A set of data collected on an x-ray diffractometer, 2theta dispersive.
    """
    peak_list = {}
    _df = None # Replaced by load_diffractogram() method
    def __init__(self, filename=None, material=None, *args, **kwargs):
        self.material = material
        self.filename=filename
        file_exists = filename is not None and os.path.isfile(filename)
        if file_exists:
            self.load_diffractogram(filename)
        self.cached_data = {}

    def diffractogram(self, filename=None):
        """Return a pandas dataframe with the X-ray diffractogram for this
        scan.
        """
        if not filename is None:
            df = self.load_diffractogram(filename)
        else:
            df = self._df
        return df

    def load_diffractogram(self, filename):
        self._df = pd.read_csv(filename, names=['2theta', 'counts'],
                         sep=' ', comment='!', index_col=0)
        self.subtract_background()
        return self._df

    def subtract_background(self):
        """
        Calculate the baseline for the diffractogram and generate a
        background correction.
        """
        background = self._df.copy()
        # Remove registered peaks
        for key, peak in self.material.peak_list.items():
            background.drop(background[peak[0]:peak[1]].index, inplace=True)
        # Determine a background line from the noise without peaks
        self.spline = UnivariateSpline(
            x=background.index,
            y=background.counts,
            s=len(background.index)*25,
            k=4
        )
        # Extrapolate the background for the whole spectrum
        x = self._df.index
        self._df['background'] = self.spline(x)
        self._df['subtracted'] = self._df.counts - self._df.background
        return self._df

    def plot_diffractogram(self, ax=None):

        """
        Plot the XRD diffractogram for this scan. Generates a new set of axes
        unless supplied by the `ax` keyword.
        """
        df = self.diffractogram()
        if not ax:
            fig = pyplot.figure()
            ax = pyplot.gca()
        ax.plot(df.index, df.loc[:, 'counts'])
        ax.plot(df.index, self.spline(df.index))
        # Highlight peaks of interest
        peak_indexes = []
        self.material.highlight_peaks(ax)
        # Set plot annotations
        ax.set_xlim(left=df.index.min(), right=df.index.max())
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r'$2\theta$')
        ax.set_ylabel('Counts')
        ax.set_title(self.axes_title())
        return ax

    def axes_title(self):
        if self.filename is None:
            title = "XRD Diffractogram"
        else:
            title = self.filename
        return title
