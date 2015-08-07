# -*- coding: utf-8 -*-

import os

import numpy as np
from scipy.interpolate import UnivariateSpline
import pandas as pd
from matplotlib import pyplot

import exceptions
import plots
from filters import fourier_transform
from xrd.tube import tubes
from xrd.peak import remove_peak_from_df
import adapters

def align_scans(scan_list, peak):
    """Align each scan to the peak in the given range. The first scan in
    the list is not modified."""
    # First calculate the two-theta position to use as a reference.
    referenceScan = scan_list[0]
    referencePeak = referenceScan.peak_position(peak)
    for scan in scan_list[1:]:
        scanPeak = scan.peak_position(peak)
        offset = referencePeak - scanPeak
        scan.shift_diffractogram(offset)
    return scan_list

class XRDScan():
    """
    A set of data collected on an x-ray diffractometer, 2theta dispersive.
    """
    _df = None # Replaced by load_diffractogram() method
    diffractogram_is_loaded = False
    spline = None
    filename = None
    def __init__(self, filename=None, name=None,
                 material=None, tube='Cu', wavelength=None):
        self.material = material
        self.cached_data = {}
        # Determine wavelength from tube type
        self.tube = tubes[tube]
        self.wavelength = self.tube.kalpha
        # Load diffractogram from file
        self.name = name
        if filename is not None:
            self.filename=filename
            self.load_diffractogram(filename)

    def __repr__(self):
        return "<{cls}: {filename}>".format(cls=self.__class__.__name__,
                                            filename=self.filename)

    @property
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
        # Prepare adapter for importing the file
        ADAPTERS = {
            '.plt': adapters.BrukerPltFile,
            '.xye': adapters.BrukerXyeFile,
            '.brml': adapters.BrukerBrmlFile
        }
        try:
            Adapter = ADAPTERS[extension]
        except KeyError:
            # Unknown file format, raise exception
            msg = 'Unknown file format {}.'.format(extension)
            raise exceptions.FileFormatError(msg)
        else:
            adapter = Adapter(filename)
        df = adapter.dataframe
        self.name = adapter.sample_name
        # Select only the two-theta range of interest
        if self.material:
            rng = self.material.two_theta_range
            df = df.loc[rng[0]:rng[1]]
        # Store dataframe and set flags
        self._df = df
        self.diffractogram_is_loaded = True
        # Subtract background
        if self.material is not None:
            self.subtract_background()
        return self._df

    def subtract_background(self):
        """
        Calculate the baseline for the diffractogram and generate a
        background correction.
        """
        background = self.diffractogram.copy()
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

    @property
    def has_background(self):
        """Returns true if the background has been fit and subtracted in the
        dataframe.
        """
        hasBackground = 'background' in self.diffractogram.columns
        return hasBackground

    def shift_diffractogram(self, offset):
        """Slide the whole diffractogram to the right by offset."""
        df = self.diffractogram
        df['2theta'] = df.index + offset
        df.set_index('2theta', inplace=True)

    def plot_diffractogram(self, ax=None):
        """
        Plot the XRD diffractogram for this scan. Generates a new set of axes
        unless supplied by the `ax` keyword.
        """
        df = self.diffractogram
        if ax is None:
            ax = plots.big_axes()
        ax.plot(df.index, df.loc[:, 'counts'])
        if self.has_background:
            ax.plot(df.index, self.spline(df.index))
        # Highlight peaks of interest
        if self.material is not None:
            self.material.highlight_peaks(ax=ax)
        # Plot fitted peaks
        if self.material is not None:
            for phase in self.material.phase_list:
                for peak in phase.peak_list:
                    peak.plot_overall_fit(ax=ax, background=self.spline)
        # Set plot annotations
        ax.set_xlim(left=df.index.min(), right=df.index.max())
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r'$2\theta$')
        ax.set_ylabel('Counts')
        ax.set_title(self.axes_title())
        return ax

    def axes_title(self):
        if self.name is not None:
            title = self.name
        elif self.filename is not None:
            title = self.filename
        else:
            title = "XRD Diffractogram"
        return title

    def fourier_transform(self):
        df = self.diffractogram
        newData = fourier_transform(pd.Series(data=df.counts, index=df.index))
        return newData

    def plot_fourier_transform(self, ax=None):
        """Perform a fourier transform on the origina data and plot"""
        if ax is None:
            ax = plots.new_axes()
        # Perform fourier transform
        fData = self.fourier_transform()
        # Plot results
        ax.plot(fData.index, fData.values,
                marker='.', linestyle='None')
        ax.set_xscale('log')
        ax.set_ylabel('Amplitude')
        ax.set_xlabel('$Frequency\ /deg^{-1}$')
        ax.set_title('Fourier Transform of {}'.format(self.axes_title()))
        ax.set_xlim(right=fData.index.max())
        return ax

    def plot_noise_reduction(self, noise_filter):
        # Generate data
        originalData = self.diffractogram.counts
        newData = noise_filter.apply(originalData)
        diff = noise_filter.difference(originalData)
        # Plot data
        fig, axArray = pyplot.subplots(3, sharex=True)
        ax1, ax2, ax3 = axArray
        ax1.plot(originalData)
        ax1.set_title('Original Diffractogram')
        ax2.plot(newData)
        ax2.set_title('New Diffractogram')
        ax3.plot(diff)
        ax3.set_title('Difference')
        # axMin, axMax = ax2.get_ylim()
        # ax3.set_ylim(-(axMax-axMin)/2,(axMax-axMin)/2)

    def contains_peak(self, two_theta_range):
        """Does this instance have the given peak within its two_theta_range?"""
        df = self.diffractogram
        two_theta_max = df.index.max()
        two_theta_min = df.index.min()
        isInRange = (two_theta_min < two_theta_range[0] < two_theta_max
                     or two_theta_min < two_theta_range[1] < two_theta_max)
        return isInRange

    @property
    def peak_list(self):
        """Return a list of all observed peaks for which the phases have been
        indexed."""
        peak_list = []
        # Look through phases and collect peaks
        for phase in self.material.phase_list:
            peak_list += phase.peak_list
        return set(peak_list)

    def peak_area(self, two_theta_range):
        """Integrated area for the given peak."""
        fullDF = self.diffractogram
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
        fullDF = self.diffractogram
        if self.has_background: column = 'subtracted'
        else: column = 'counts'
        peakDF = fullDF.loc[
            twotheta_range[0]:twotheta_range[1],
            column
        ]
        twotheta = peakDF.argmax()
        return twotheta

    def peak_fwhm(self, two_theta_range):
        """Calculate the full-width half max."""
        fullDF = self.diffractogram
        peakDF = fullDF.loc[
            two_theta_range[0]:two_theta_range[1],
            'subtracted'
        ]
        maxIdx = peakDF.argmax()
        maxCounts = peakDF[maxIdx]
        # Split dataframe into left and right
        leftDF = peakDF.loc[two_theta_range[0]:maxIdx]
        rightDF = peakDF.loc[maxIdx:two_theta_range[1]]
        # Find the data points around the half-max
        def half_max(df, maximum):
            """Return index of half-maximum of dataframe."""
            halfMax = maximum/2
            upperDf = df[df > halfMax]
            lowerDf = df[df < halfMax]
            if upperDf.empty or lowerDf.empty:
                print("Cannot compute FWHM at {}".format(self.cube_coords))
                newIdx = df.argmax()
            else:
                upperIdx = df[df > halfMax].argmin()
                upperVal = df[upperIdx]
                lowerIdx = df[df < halfMax].argmax()
                lowerVal = df[lowerIdx]
                # Interpolate between the two data points
                slope = (upperVal - lowerVal)/(upperIdx - lowerIdx)
                newIdx = (halfMax - lowerVal)/slope + lowerIdx
            return newIdx
        leftIdx = half_max(leftDF, maxCounts)
        rightIdx = half_max(rightDF, maxCounts)
        fullWidth = rightIdx - leftIdx
        return fullWidth

    def fit_peaks(self):
        for phase in self.material.phase_list:
            phase.fit_peaks(scan=self)

    def refine_unit_cells(self):
        """Residual least-squares refinement of the unit cell for each
        phase. Warning: overlapping reflections from different phases is
        likely to cause considerable errors."""
        raise NotImplementedError
