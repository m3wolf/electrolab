# -*- coding: utf-8 -*-

import os
from collections import namedtuple

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
    def __init__(self, reflection_list=[], diagnostic_reflection=None, unit_cell=None, name="phase", space_group=None):
        self.reflection_list = reflection_list
        self.name = name
        self.space_group = space_group
        if diagnostic_reflection is not None:
            self.diagnostic_reflection = self.reflection_by_hkl(diagnostic_reflection)
        else:
            self.diagnostic_reflection = Reflection()

    def reflection_by_hkl(self, hkl_input):
        for reflection in self.reflection_list:
            if reflection.hkl == hkl_to_tuple(hkl_input):
                return reflection

    def predicted_peak_positions(self):
        pass

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.name)


class UnitCellError(ValueError):
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
        # Set initial cell parameters
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

    class FixedAngle():
        """A Unit-cell angle that cannot change for that unit cell"""
        def __init__(self, angle, name='angle'):
            self.angle = angle
            self.name = name

        def __get__(self, obj, objtype):
            return angle

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


class HexagonalUnitCell(UnitCell):
    """Unit cell where a=b, α=β=90°, γ=120°."""
    free_parameters = ('a', 'c')
    a = UnitCell.ConstrainedLength()
    b = UnitCell.ConstrainedLength()
    alpha = UnitCell.FixedAngle(angle=90, name="α")
    beta = UnitCell.FixedAngle(angle=90, name="β")
    gamma = UnitCell.FixedAngle(angle=120, name="γ")


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
        df = self.diffractogram()
        # Make a list of all reflections
        reflection_list = []
        for phase in self.material.phase_list:
            reflection_list += phase.reflection_list
        # Find two_theta of each reflection
        peak_list = []
        for reflection in reflection_list:
            peakPosition = self.peak_position(reflection.two_theta_range)
            peak_list.append(peakPosition)
        return peak_list

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

    @property
    def cell_parameters(self):
        # - Calculate predicted two-theta peaks from space group and unit-cell parameters
	# - Compare predicted and experimental peak positions
	# - Change the unit-cell parameters
	# - Recalculate the mean-square-difference
	# - Keep repeating until a minimum is reached
        pass

    def refine_cell_parameters(self):
        """Residual least-squares refinement of the unit cell."""
        pass
