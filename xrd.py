# -*- coding: utf-8 -*-

import math
import os
from collections import namedtuple

import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot

from filters import fourier_transform, LowPassFilter, HighPassFilter
from xrdpeak import tubes, Peak
import plots
import exceptions
import adapters

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
        df = scan.diffractogram
        df.counts = df.counts + step_size * idx
        ax.plot(df.index, df.counts)
        filenames.append(scan.filename)
    ax.legend(filenames)
    # Decorate
    ax.set_xlabel(r'$2\theta$')
    ax.set_ylabel('counts')

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


HKL = namedtuple('HKL', ('h', 'k', 'l'))

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
        hkl_tuple = HKL(self._h, self._k, self._l)
        return hkl_tuple

    @property
    def hkl_string(self):
        string = "{h}{k}{l}".format(h=self._h, k=self._k, l=self._l)
        return string

    def __repr__(self):
        template = "<Reflection: {0}{1}{2}>"
        return template.format(self._h, self._k, self._l)

    def __str__(self):
        template = "({0}{1}{2})"
        return template.format(self._h, self._k, self._l)


class Phase():
    """A crystallographic phase that can be found in a Material."""
    reflection_list = [] # Predicted peaks by crystallography

    def __repr__(self):
        print(self.name.__class__)
        return "<{}: {}>".format(self.__class__.__name__, self.name)

    def reflection_by_hkl(self, hkl_input):
        for reflection in self.reflection_list:
            if reflection.hkl == hkl_to_tuple(hkl_input):
                return reflection

    @property
    def diagnostic_reflection(self):
        reflection = self.reflection_by_hkl(self.diagnostic_hkl)
        return reflection

    def refine_unit_cell(self, scan, quiet=False):
        """Residual least squares refinement of the unit-cell
        parameters. Returns the residual root-mean-square error between
        predicted and observed 2θ."""
        # Define an objective function that will be minimized
        def objective(cell_parameters):
            # Determine cell parameters from numpy array
            # Create a temporary unit cell and return the residual error
            unit_cell = self.unit_cell.__class__()
            unit_cell.set_cell_parameters_from_list(cell_parameters)
            residuals = self.peak_rms_error(scan=scan,
                                            unit_cell=unit_cell)
            return residuals
        # Fit peaks to Gaussian/Cauchy functions using least squares refinement
        self.fit_peaks(scan=scan)
        # Now minimize it
        initial_parameters = self.unit_cell.cell_parameters
        result = scipy.optimize.minimize(fun=objective,
                                         x0=initial_parameters,
                                         method='Nelder-Mead',
                                         options={'disp': not quiet})
        if result.success:
            # Optimiziation was successful, so set new parameters
            optimized_parameters = result.x
            self.unit_cell.set_cell_parameters_from_list(optimized_parameters)
            residual_error = self.peak_rms_error(scan=scan)
            return residual_error
        else:
            # Optimization failed for some reason
            raise RefinementError(result.message)

    def predicted_peak_positions(self, wavelength, unit_cell=None):
        # Use current unit_cell if none is given
        if unit_cell is None:
            unit_cell = self.unit_cell
        PredictedPeak = namedtuple('PredictedPeak', ('hkl', 'd', 'two_theta'))
        predicted_peaks = []
        for reflection in self.reflection_list:
            hkl = reflection.hkl
            d = unit_cell.d_spacing(hkl)
            radians = math.asin(wavelength/2/d)
            two_theta = 2*math.degrees(radians)
            predicted_peaks.append(
                PredictedPeak(reflection.hkl_string, d, two_theta)
            )
        return predicted_peaks

    @property
    def peak_list(self):
        peak_list = getattr(self, '_peak_list', None)
        if peak_list is None:
            # Peak fitting has not been performed, raise and error
            msg = 'Peak fitting has not been performed. Please run {cls}.fit_peaks method'
            print(msg.format(cls=self.__class__.__name__))
            # raise exceptions.PeakFitError(msg.format(cls=self.__class__.__name__))
            peak_list = []
        return peak_list

    def fit_peaks(self, scan):
        """
        Use least squares refinement to fit gaussian/Cauchy/etc functions
        to the predicted reflections.
        """
        self._peak_list = []
        fitMethods = ['pseudo-voigt', 'gaussian', 'cauchy', 'estimated']
        for reflection in self.reflection_list:
            if scan.contains_peak(reflection.two_theta_range):
                newPeak = Peak(reflection=reflection)
                df = scan.diffractogram.loc[
                    reflection.two_theta_range[0]:reflection.two_theta_range[1]
                ]
                # Try each fit method until one works
                for method in fitMethods:
                    try:
                        residual = newPeak.fit(df.index, df.subtracted, method=method)
                    except exceptions.PeakFitError:
                        # Try next fit
                        continue
                    else:
                        self._peak_list.append(newPeak)
                        break
                else:
                    # No sucessful fit could be found.
                    msg = "peak could not be fit for {}.".format(reflection)
                    print(msg)
        return self.peak_list

    def peak_rms_error(self, scan, unit_cell=None):
        diffs = []
        wavelength = scan.wavelength
        predicted_peaks = self.predicted_peak_positions(wavelength=wavelength,
                                                        unit_cell=unit_cell)
        # Prepare list of peak position differences
        for idx, actual_peak in enumerate(self.peak_list):
            actual = actual_peak.center_kalpha
            predicted = predicted_peaks[idx].two_theta
            diffs.append(actual-predicted)
        # Calculate mean-square-difference
        running_total = 0
        for diff in diffs:
            running_total += diff**2
        rms_error = math.sqrt(running_total/len(diffs))
        return rms_error


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
        self._df = df
        self.diffractogram_is_loaded = True
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

    def plot_fourier_transform(self, ax=None):
        """Perform a fourier transform on the origina data and plot"""
        if ax is None:
            ax = plots.new_axes()
        df = self.diffractogram
        # Perform fourier transform
        newData = fourier_transform(pd.Series(data=df.counts, index=df.index))
        # Plot results
        ax.plot(newData.index, newData.values,
                marker='.', linestyle='None')
        ax.set_xscale('log')
        ax.set_ylabel('Amplitude')
        ax.set_xlabel('$Frequency\ /deg^{-1}$')
        ax.set_title('Fourier Transform of {}'.format(self.axes_title()))
        ax.set_xlim(right=newData.index.max())
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
        peakDF = fullDF.loc[
            twotheta_range[0]:twotheta_range[1],
            'subtracted'
        ]
        twotheta = peakDF.argmax()
        return twotheta

    def fit_peaks(self):
        for phase in self.material.phase_list:
            phase.fit_peaks(scan=self)

    def refine_unit_cells(self):
        """Residual least-squares refinement of the unit cell for each
        phase. Warning: overlapping reflections from different phases is
        likely to cause considerable errors."""
        raise NotImplementedError
