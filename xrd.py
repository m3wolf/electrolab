# -*- coding: utf-8 -*-

import math
import os
from collections import namedtuple

import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import UnivariateSpline
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

def align_scans(scan_list, peak):
    """Align each scan to the peak in the given range. The first scan in
    the list is not modified."""
    # First calculate the two-theta position to use as a reference.
    referenceScan = scan_list[0]
    referencePeak = referenceScan.peak_position(peak)
    for scan in scan_list[1:]:
        scanPeak = scan.peak_position(peak)
        offset = referencePeak - scanPeak
        scan.offset_diffractogram(offset)
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
    reflection_list = []
    def __init__(self, reflection_list=[], diagnostic_reflection=None, unit_cell=None, name="phase", space_group=None):
        self.reflection_list = reflection_list
        self.name = name
        self.unit_cell = unit_cell
        self.space_group = space_group
        if diagnostic_reflection is not None:
            self.diagnostic_reflection = self.reflection_by_hkl(diagnostic_reflection)
        else:
            self.diagnostic_reflection = Reflection()

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.name)

    def reflection_by_hkl(self, hkl_input):
        for reflection in self.reflection_list:
            if reflection.hkl == hkl_to_tuple(hkl_input):
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

    def actual_peak_positions(self, scan):
        ActualPeak = namedtuple('ActualPeak', ('hkl', 'two_theta'))
        df = scan.diffractogram()
        # Make a list of all reflections
        reflection_list = self.reflection_list
        # Find two_theta of each reflection
        peak_list = []
        for reflection in reflection_list:
            peakPosition = scan.peak_position(reflection.two_theta_range)
            peak_list.append(ActualPeak(reflection.hkl_string, peakPosition))
        return peak_list

    def peak_rms_error(self, scan, unit_cell=None):
        diffs = []
        wavelength = scan.wavelength
        actual_peaks = self.actual_peak_positions(scan=scan)
        predicted_peaks = self.predicted_peak_positions(wavelength=wavelength,
                                                        unit_cell=unit_cell)
        # Prepare list of peak position differences
        for idx, actual_peak in enumerate(actual_peaks):
            actual = actual_peak.two_theta
            predicted = predicted_peaks[idx].two_theta
            diffs.append(actual-predicted)
        # Calculate mean-square-difference
        running_total = 0
        for diff in diffs:
            running_total += diff**2
        rms_error = math.sqrt(running_total/len(diffs))
        return rms_error


class UnitCellError(ValueError):
    pass

class RefinementError(Exception):
    pass

class UnitCell():
    """Describes a crystallographic unit cell for XRD Refinement. Composed
    of up to three lengths (a, b and c) in angstroms and three angles
    (alpha, beta, gamma) in degrees. Subclasses with high symmetry
    with have less than six parameters.
    """
    free_parameters = ('a', 'b', 'c', 'alpha', 'beta', 'gamma')
    a = 1
    b = 1
    c = 1
    constrained_length = 1
    alpha = 90
    beta = 90
    gamma = 90
    def __init__(self, a=None, b=None, c=None,
                 alpha=None, beta=None, gamma=None):
        # Set initial cell parameters.
        # This method avoids setting constrained values to defaults
        for attr in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']:
            value = locals()[attr]
            if value is not None:
                self.__setattr__(attr, value)

    def __setattr__(self, name, value):
        """Check for reasonable value for crystallography parameters"""
        # Unit cell lengths
        if name in ['a', 'b', 'c'] and value <= 0:
            msg = 'unit-cell dimensions must be greater than 0 ({}={})'
            raise UnitCellError(msg.format(name, value))
        # Unit cell angles
        elif name in ['alpha', 'beta', 'gamma'] and not (0 < value < 180):
            msg = 'unit-cell angles must be between 0° and 180° ({}={}°)'
            raise UnitCellError(msg.format(name, value))
        # No problems, so set the attribute as normal
        else:
            super(UnitCell, self).__setattr__(name, value)

    def __repr__(self):
        name = '<{cls}: a={a}, b={b}, c={c}, α={alpha}, β={beta}, γ={gamma}>'
        name = name.format(cls=self.__class__.__name__,
                           a=self.a, b=self.b, c=self.c,
                           alpha=self.alpha, beta=self.beta, gamma=self.gamma)
        return name

    @property
    def cell_parameters(self):
        """Named tuple of the cell parameters that aren't fixed."""
        params = self.free_parameters
        CellParameters = namedtuple('CellParameters', params)
        paramArgs = {param: getattr(self, param) for param in params}
        parameters = CellParameters(**paramArgs)
        return parameters

    def set_cell_parameters_from_list(self, parameters_list):
        """
        Accept a list of parameters and assumes they are in the order of
        self.free_parameters. This is a shaky assumption and should not be
        used if avoidable. This method was created for use in cell refinement
        where scipy.optimize.minimize passes a numpy array.
        """
        for idx, key in enumerate(self.free_parameters):
            setattr(self, key, parameters_list[idx])

    class FixedAngle():
        """A Unit-cell angle that cannot change for that unit cell"""
        def __init__(self, angle, name='angle'):
            self.angle = angle
            self.name = name

        def __get__(self, obj, objtype):
            return self.angle

        def __set__(self, obj, value):
            msg = "{name} must equal {angle}° for {cls}"
            msg = msg.format(name=self.name,
                             angle=self.angle,
                             cls=obj.__class__.__name__)
            raise UnitCellError(msg)

    class ConstrainedLength():
        """
        Unit-cell angle that is tied to another length in the cell. Eg. a=b
        """
        def __get__(self, obj, objtype):
            return obj.constrained_length

        def __set__(self, obj, value):
            obj.constrained_length = value


class CubicUnitCell(UnitCell):
    """Unit cell where a=b=c and α=β=γ=90°"""
    free_parameters = ('a', )
    # Descriptors for unit-cell lengths, since a=b=c
    a = UnitCell.ConstrainedLength()
    b = UnitCell.ConstrainedLength()
    c = UnitCell.ConstrainedLength()
    alpha = UnitCell.FixedAngle(90, name="α")
    beta = UnitCell.FixedAngle(90, name="β")
    gamma = UnitCell.FixedAngle(90, name="γ")

    def d_spacing(self, hkl):
        """Determine d-space for the given hkl plane."""
        h, k, l = hkl
        inverse_d_squared = (h**2 + k**2 + l**2)/(self.a**2)
        d = math.sqrt(1/inverse_d_squared)
        return d

class HexagonalUnitCell(UnitCell):
    """Unit cell where a=b, α=β=90°, γ=120°."""
    free_parameters = ('a', 'c')
    a = UnitCell.ConstrainedLength()
    b = UnitCell.ConstrainedLength()
    alpha = UnitCell.FixedAngle(angle=90, name="α")
    beta = UnitCell.FixedAngle(angle=90, name="β")
    gamma = UnitCell.FixedAngle(angle=120, name="γ")

    def d_spacing(self, hkl):
        """Determine d-space for the given hkl plane."""
        h, k, l = hkl
        a, c = (self.a, self.c)
        inverse_d_squared = 4*(h**2 + h*k + k**2)/(3*a**2) + l**2/(c**2)
        d = math.sqrt(1/inverse_d_squared)
        return d


tube_wavelengths = {
    'Cu': 1.5418
}

class XRDScan():
    """
    A set of data collected on an x-ray diffractometer, 2theta dispersive.
    """
    _df = None # Replaced by load_diffractogram() method
    diffractogram_is_loaded = False
    filename = None
    def __init__(self, filename=None, material=None, tube='Cu', wavelength=None):
        self.material = material
        self.cached_data = {}
        # Determine wavelength from tube type
        if wavelength is None:
            self.wavelength = tube_wavelengths[tube]
        else:
            self.wavelength = wavelength
        # Load diffractogram from file
        if filename is not None:
            self.filename=filename
            self.load_diffractogram(filename)

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

    def offset_diffractogram(self, offset):
        """Slide the whole diffractogram to the right by offset."""
        df = self.diffractogram()
        df['2theta'] = df.index + offset
        df.set_index('2theta', inplace=True)

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

    @property
    def peak_list(self):
        """Return a list of all observed peaks for which the phases have been
        indexed."""
        peak_list = []
        # Look through phases and collect peaks
        for phase in self.material.phase_list:
            for peak in phase.actual_peak_positions(scan=self):
                peak_list.append(peak.two_theta)
        return set(peak_list)

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

    def refine_unit_cells(self):
        """Residual least-squares refinement of the unit cell for each
        phase. Warning: overlapping reflections from different phases is
        likely to cause considerable errors."""
        raise NotImplementedError
