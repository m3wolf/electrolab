# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import scipy
from matplotlib import pyplot


def hkl_to_tuple(hkl_input):
    """If hkl_input is a string, extract the hkl values and
    return them as (h, k, l). If hkl_string is not a string, return it
    unmodified."""
    # Convert hkl to tuple dependent on form
    hklTuple = None
    if isinstance(hkl_input, tuple):
        # Already a tuple, no action
        hklTuple = hkl_input
    elif isinstance(hkl_input, str):
        # String to tuple
        hklTuple = (
            int(hkl_input[0]),
            int(hkl_input[1]),
            int(hkl_input[2])
        )
    return hklTuple

def remove_peak_from_df(reflection, df):
    """Accept an xrd scan dataframe and remove the given reflection's peak from
    the data."""
    peak = reflection.two_theta_range
    df.drop(df[peak[0]:peak[1]].index, inplace=True)

def plot_scans(scan_list, step_size=1):
    """Plot a series of XRDScans as a waterfall."""
    fig = pyplot.figure(figsize=(16, 9))
    ax = pyplot.gca()
    filenames = []
    for idx, scan in enumerate(scan_list):
        df = scan.diffractogram()
        df.counts = df.counts + step_size * idx
        ax.plot(df.index, df.counts)
        filenames.append(scan.filename)
    ax.legend(filenames)
    # Decorate
    ax.set_xlabel(r'$2\theta$')
    ax.set_ylabel('counts')

class Reflection():
    """An XRD reflection with a specific hkl value."""
    def __init__(self, two_theta_range=(10, 80), hkl=(0, 0, 0)):
        self.two_theta_range = two_theta_range
        h, k, l = hkl_to_tuple(hkl)
        self._h = h
        self._k = k
        self._l = l

    @property
    def hkl(self):
        return (self._h, self._k, self._l)

    def __repr__(self):
        template = "<Reflection: {0}{1}{2}>"
        return template.format(self._h, self._k, self._l)

    def __str__(self):
        template = "({0}{1}{2})"
        return template.format(self._h, self._k, self._l)


class Phase():
    """A crystallographic phase that can be found in a Material."""
    reflection_list = []
    def __init__(self, reflection_list=[], diagnostic_reflection=None, crystal_system=None, name="phase"):
        self.reflection_list = reflection_list
        self.name = name
        if diagnostic_reflection is not None:
            self.diagnostic_reflection = self.reflection_by_hkl(diagnostic_reflection)
        else:
            self.diagnostic_reflection = Reflection()

    def reflection_by_hkl(self, hkl_input):
        for reflection in self.reflection_list:
            if reflection.hkl == hkl_to_tuple(hkl_input):
                return reflection

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.name)

class XRDScan():
    """
    A set of data collected on an x-ray diffractometer, 2theta dispersive.
    """
    _df = None # Replaced by load_diffractogram() method
    diffractogram_is_loaded = False
    filename = None
    def __init__(self, filename=None, material=None, *args, **kwargs):
        self.material = material
        if filename is not None:
            self.filename=filename
            self.load_diffractogram(filename)
        self.cached_data = {}

    def diffractogram(self, filename=None):
        """Return a pandas dataframe with the X-ray diffractogram for this
        scan.
        """
        if filename is None:
            df = self._df
        else:
            df = self.load_diffractogram(filename)
        return df

    def load_diffractogram(self, filename):
        # Determine file type from extension
        fileBase, extension = os.path.splitext(filename)
        if extension == '.xye':
            csvKwargs = {
                'sep': ' ',
                'index_col': 0,
                'names': ['2theta', 'counts', 'error'],
                'comment': "'",
            }
        elif extension == '.plt':
            csvKwargs = {
                'sep': ' ',
                'index_col': 0,
                'names': ['2theta', 'counts'],
                'comment': '!',
            }
        else:
            # Unknown file format, guess anyway
            msg = 'Unknown file format {}.'.format(extension)
            raise ValueError(msg)
        self._df = pd.read_csv(filename, **csvKwargs)
        self.diffractogram_is_loaded = True
        self.subtract_background()
        return self._df

    def subtract_background(self):
        """
        Calculate the baseline for the diffractogram and generate a
        background correction.
        """
        background = self.diffractogram().copy()
        # Remove pre-indexed peaks for background fitting
        if self.material is not None:
            phase_list = self.material.phase_list + self.material.background_phases
            for phase in phase_list:
                for reflection in phase.reflection_list:
                    remove_peak_from_df(reflection, background)
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
        self.material.highlight_peaks(ax=ax)
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

    def contains_peak(self, peak):
        """Does this instance have the given peak within its two_theta_range?"""
        df = self.diffractogram()
        two_theta_max = df.index.max()
        two_theta_min = df.index.min()
        isInRange = (two_theta_min < peak[0] < two_theta_max
                     or two_theta_min < peak[1] < two_theta_max)
        return isInRange

    def peak_area(self, two_theta_range):
        """Integrated area for the given peak."""
        fullDF = self.diffractogram()
        # Get peak dataframe for integration
        if self.contains_peak(two_theta_range):
            peakDF = fullDF.loc[
                two_theta_range[0]:two_theta_range[1],
                'subtracted'
            ]
            # Integrate peak
            area = np.trapz(x=peakDF.index, y=peakDF)
        else:
            area = 0
        return area

    def peak_position(self, twotheta_range):
        fullDF = self.diffractogram()
        peakDF = fullDF.loc[
            twotheta_range[0]:twotheta_range[1],
            'subtracted'
        ]
        twotheta = peakDF.argmax()
        return twotheta
