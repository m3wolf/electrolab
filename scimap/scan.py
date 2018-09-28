# -*- coding: utf-8 -*-

from functools import lru_cache

import pandas as pd
from matplotlib import pyplot

from . import plots
from .filters import fourier_transform
from .tube import tubes
from .native_refinement import NativeRefinement
from .adapters import adapter_from_filename
from .utilities import q_to_twotheta


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
    """A set of data collected on an x-ray diffractometer or synchrotron
    beamline.

    Parameters
    ----------
    filename : str
      Path to the data file to load
    name : str, optional
      Human-readable name to use for plot legends etc.
    phases : list, optional
      Instances of the ``Phase`` class to use for refinement.
    background_phases : list, optional
      Instances of the ``Phase`` class to ignore during refinement
    tube : str, optional
      X-ray tube material, this will be used to determine the
      wavelength if not explicitly given by the ``wavelength`` argument.
    wavelength : float, optional
      Radiation wavelength in angstroms. If given, this will override
      the ``tube`` argument.
    Refinement : optional
      A subclass of ``BaseRefinement``, it will be used for refining
      the XRD data.

    """
    _df = None  # Replaced by load_diffractogram() method
    diffractogram_is_loaded = False
    spline = None
    filename = None

    def __init__(self, filename='', name=None,
                 phases=[], phase=None, background_phases=[],
                 tube='Cu', wavelength=None,
                 Refinement=NativeRefinement, two_theta_range=None):
        self._filename = filename
        if phase is not None:
            self.phases = [phase]
        else:
            self.phases = phases
        self.background_phases = background_phases
        self.cached_data = {}
        self.refinement = Refinement(phases=self.phases)
        self._two_theta_range = two_theta_range
        # Determine wavelength from tube type
        if wavelength is not None:
            self.wavelength = wavelength
        else:
            self.tube = tubes[tube]
            self.wavelength = self.tube.kalpha
        # Load diffractogram from file
        self.name = name

    def __repr__(self):
        return "<{cls}: {filename}>".format(cls=self.__class__.__name__,
                                            filename=self.filename)

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, val):
        self._filename = val

    @property
    def scattering_lengths(self):
        with adapter_from_filename(self.filename) as f:
            return f.scattering_lengths(wavelength=self.wavelength)

    @property
    def two_theta(self):
        with adapter_from_filename(self.filename) as f:
            return f.two_theta()

    @property
    def intensities(self):
        with adapter_from_filename(self.filename) as f:
            return f.intensities()

    @property
    def diffractogram(self):
        """Return a pandas dataframe with the X-ray diffractogram for this
        scan.
        """
        data = {
            'counts': self.intensities,
            # 'subtracted': self.intensities - self.background(),
        }
        q = self.scattering_lengths
        df = pd.DataFrame(index=q, data=data)
        return df

    # @diffractogram.setter
    # def diffractogram(self, new_df):
    #     self._df = new_df

    def save_diffractogram(self, filename):
        # Determine file type from extension
        adapter = adapter_from_filename(filename)
        result = adapter.write_diffractogram(scan=self)
        return result

    def _load_diffractogram(self, filename):
        adapter = adapter_from_filename(filename)
        df = adapter.dataframe
        self.name = adapter.sample_name
        # Select only the two-theta range of interest
        if self._two_theta_range is not None:
            rng = self._two_theta_range
            df = df.loc[rng[0]:rng[1]]
        # Store dataframe and set flags
        self._df = df
        self.diffractogram_is_loaded = True
        return self._df

    @property
    def raw_data(self):
        """Return the imported XRD pattern as a 2-tuple of (scattering_length,
        intensity) data.
        """
        return self.scattering_lengths, self.intensities

    @lru_cache()
    def background(self):
        """Return background that has been refined."""
        q, I = self.raw_data
        self.refinement.refine_background(q, I, s=len(q)*25)
        background = self.refinement.background(q)
        return background

    def shift_diffractogram(self, offset):
        """Slide the whole diffractogram to the right by offset."""
        df = self.diffractogram
        df['2theta'] = df.index + offset
        df.set_index('2theta', inplace=True)

    def plot_diffractogram(self, ax=None, marker='+',
                           linestyle=":", use_twotheta=False,
                           wavelength=None, *args, **kwargs):
        """Plot the XRD diffractogram for this scan. Generates a new set of
        axes unless supplied by the `ax` keyword.

        Arguments
        ---------
        - ax : A matplotlib axes that will receive the plotted data

        - marker : A string for how to plot each datum. Similar spec
        to matplotlib.

        - linestyle : A matplotlib linestyle spec for how to connect the dots.

        - use_twotheta : If truthy, the plot will convert scattering
          lengths to 2θ before plotting. If used, the `wavelength`
          argument must also be given.

        - wavelength : Wavelength of light to be used for converstion
          to two_theta. If wavelength is `None` (default), the scan
          wavelength will be used.

        """
        q = self.scattering_lengths
        # Set default radiation wavelength
        if wavelength is None:
            wavelength = self.wavelength
        # Convert q to 2θ if requested
        if use_twotheta:
            x = q_to_twotheta(q, wavelength=wavelength)
        else:
            x = q
        y = self.intensities
        if ax is None:
            ax = plots.xrd_axes()
        ax.set_xlim(left=x.min(), right=x.max())
        ax.plot(x, y, marker=marker, linestyle=linestyle, *args, **kwargs)

        # Set plot annotations
        if use_twotheta:
            ax.xaxis.set_major_formatter(plots.DegreeFormatter())
            ax.set_xlabel(r'$2\theta$')
        else:
            ax.set_xlabel(r'q /$A^{-}$')
        ax.set_ylabel('Counts')
        ax.set_title(self.axes_title())
        return ax

    def plot_diffractogram_2d(self, ax=None, mask="auto"):
        if ax is None:
            ax = plots.new_image_axes()
        adapter = adapter_from_filename(self.filename)
        img = adapter.detector_image()
        img_ax = ax.imshow(img, cmap="viridis")
        return img_ax

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
        area = self.refinement.net_area(two_theta_range)
        return area

    def peak_position(self, twotheta_range):
        fullDF = self.diffractogram
        if self.has_background:
            column = 'subtracted'
        else:
            column = 'counts'
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
            halfMax = maximum / 2
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
                slope = (upperVal - lowerVal) / (upperIdx - lowerIdx)
                newIdx = (halfMax - lowerVal) / slope + lowerIdx
            return newIdx
        leftIdx = half_max(leftDF, maxCounts)
        rightIdx = half_max(rightDF, maxCounts)
        fullWidth = rightIdx - leftIdx
        return fullWidth
