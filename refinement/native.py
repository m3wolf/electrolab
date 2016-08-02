# -*- coding: utf-8 -*-

import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import UnivariateSpline
import pandas

import exceptions
import plots
from xrd.peak import XRDPeak
from peakfitting import remove_peak_from_df
from refinement.base import BaseRefinement
from mapping.datadict import DataDict


def contains_peak(scattering_lengths, qrange):
    """Does this instance have the given peak within its range of q values?"""
    qmax = max(scattering_lengths)
    qmin = min(scattering_lengths)
    isInRange = (qmin < qrange[0] < qmax or
                 qmin < qrange[1] < qmax)
    return isInRange


class NativeRefinement(BaseRefinement):
    data_dict = DataDict(['spline', 'is_refined'])
    spline = None

    def peak_rms_error(self, phase, unit_cell=None, peak_list=None):
        diffs = []
        predicted_peaks = phase.predicted_peak_positions(unit_cell=unit_cell)
        # Only include those that are within the two_theta range
        phase_idx = self.phases.index(phase)
        if peak_list is None:
            actual_peaks = [p.center() for p in self._peak_list[phase_idx]]
        else:
            actual_peaks = peak_list
        # Prepare list of peak position differences
        for idx, actual_peak in enumerate(actual_peaks):
            offsets = [abs(p.q-actual_peak)
                       for p in predicted_peaks]
            diffs.append(min(offsets))
        # Calculate mean-square-difference
        running_total = 0
        for diff in diffs:
            running_total += diff**2
        try:
            rms_error = math.sqrt(running_total / len(diffs))
        except ZeroDivisionError:
            raise exceptions.RefinementError()
        return rms_error

    def refine_unit_cells(self, scattering_lengths, intensities, quiet=True):
        """Residual least squares refinement of the unit-cell
        parameters. Returns an (p, 6) array where p is the number of
        phases and axis 1 has a value for each of the cell parameters
        (a, b, c, α, β, γ).
        """
        # Fit peaks to Gaussian/Cauchy functions using least squares refinement
        # self.fit_peaks(scattering_lengths=scattering_lengths,
        #                intensities=intensities)
        # Get a list of peak positions
        assert len(self.phases) == 1 # Temporary to avoid weird fitting
        peak_list = []
        for reflection in self.phases[0].reflection_list:
            if contains_peak(scattering_lengths, reflection.qrange):
                left = reflection.qrange[0]
                right = reflection.qrange[1]
                idx = np.where(np.logical_and(left < scattering_lengths,
                                              scattering_lengths < right))
                xs = scattering_lengths[idx]
                ys = intensities[idx]
                maxidx = ys.argmax()
                xmax = xs[maxidx]
                peak_list.append(xmax)
        for phase in self.phases:
            # Define an objective function that will be minimized
            def objective(cell_parameters):
                # Determine cell parameters from numpy array
                # Create a temporary unit cell and return the residual error
                unit_cell = phase.unit_cell.__class__()
                unit_cell.set_cell_parameters_from_list(cell_parameters)
                residuals = self.peak_rms_error(phase=phase,
                                                unit_cell=unit_cell,
                                                peak_list = peak_list)
                return residuals
            # Now minimize objective for each phase
            initial_parameters = phase.unit_cell.cell_parameters
            result = scipy.optimize.minimize(fun=objective,
                                             x0=initial_parameters,
                                             method='Nelder-Mead',
                                             options={'disp': not quiet})
            if result.success:
                # Optimiziation was successful, so set new parameters
                optimized_parameters = result.x
                phase.unit_cell.set_cell_parameters_from_list(
                    optimized_parameters)
                residual_error = self.peak_rms_error(phase=phase, peak_list=peak_list)
                return residual_error
            else:
                # Optimization failed for some reason
                raise exceptions.RefinementError(result.message)

    def refine_scale_factors(self):
        # Make sure background is refined first
        if not self.is_refined['background']:
            self.refine_background()
        # Scale factor is just the ratio of peak area of diagnostic reflection
        for phase in self.scan.phases:
            reflection = phase.diagnostic_reflection
            area = self.net_area(two_theta_range=reflection.two_theta_range)
            phase.scale_factor = area
        self.is_refined['scale_factors'] = True

    def refine_background(self, scattering_lengths, intensities, s=None, k=4):
        """Fit a univariate spline to the background data.

        Arguments
        ---------
        - scattering_lengths : Array of scattering vector lengths, q.

        - intensities : Array of intensity values at each q position

        - s : Smoothing factor passed to the spline. Default is
          len(scattering_lengths)

        - k : Degree of the spline (default quartic spline)
        """
        if s is None:
            s = len(scattering_lengths)
        # Remove pre-indexed peaks for background fitting
        phase_list = self.phases + self.background_phases
        q, I = scattering_lengths, intensities
        for phase in phase_list:
            for reflection in phase.reflection_list:
                q, I = remove_peak_from_df(x=q, y=I, xrange=reflection.qrange)
        # Determine a background line from the noise without peaks
        self.spline = UnivariateSpline(
            x=q,
            y=I,
            s=s / 25,
            k=3
        )
        # Extrapolate the background for the whole pattern
        background = self.spline(scattering_lengths)
        return background

    def refine_peak_widths(self):
        pass

    def fwhm(self, phase_idx=0):
        """Use the previously fitted peaks to describe full-width and
        half-maximum."""
        phase = self.scan.phases[phase_idx]
        peak = self.peak(phase.diagnostic_reflection, phase_idx=phase_idx)
        # print('TODO: Get diagnostic peak instead of', peak)
        return peak.fwhm()

    @property
    def has_background(self):
        """Returns true if the background has been fit and subtracted.
        """
        return self.spline is not None

    def background(self, x):
        if self.spline is None:
            raise exceptions.RefinementError("Please run `refine_background()` first")
        return self.spline(x)

    def refine_displacement(self):
        """Not implemented yet."""
        pass

    def net_area(self, two_theta_range):
        """Area under the scan diffractogram after background subtraction."""
        # fullDF = self.scan.diffractogram
        df = self.subtracted
        # Get peak dataframe for integration
        if self.contains_peak(scattering_lengths, two_theta_range):
            netDF = df.loc[two_theta_range[0]:two_theta_range[1]]
            # Integrate peak
            area = np.trapz(x=netDF.index, y=netDF)
        else:
            area = 0
        return area

    def details(self):
        return "Native refinement"

    def peak_list_by_phase(self):
        """List of fitted peaks organized by phase."""
        peak_list = getattr(self, '_peak_list', None)
        if peak_list is None:
            peak_list = self.fit_peaks()
        return peak_list

    @property
    def peak_list(self):
        """List of fitted peaks across all phases"""
        peak_list = self.peak_list_by_phase()
        # Flatten the list of phases/peaks
        full_list = []
        for phase in peak_list:
            full_list += phase
        return full_list

    def peak(self, reflection, phase_idx=0):
        """Returns a specific fitted peak."""
        peak_list = self.peak_list_by_phase()[phase_idx]
        peaks = [peak for peak in peak_list if peak.reflection == reflection]
        # Check for sanity
        if len(peaks) < 1:
            raise ValueError(
                'Peak for reflection {} was not found'.format(reflection)
            )
        elif len(peaks) > 1:
            raise IndexError('Mutliple peaks found for {}'.format(reflection))
        # Sanity checks passed so return to only value
        return peaks[0]

    def fit_peaks(self, scattering_lengths, intensities):
        """
        Use least squares refinement to fit gaussian/Cauchy/etc functions
        to the predicted reflections.
        """
        raise NotImplementedError("Disabled due to bad background fitting.")
        self._peak_list = []
        fitMethods = ['pseudo-voigt', 'gaussian', 'cauchy', 'estimated']
        reflection_list = []
        # Check if there are phases present
        if len(self.phases) == 0:
            msg = '{this} has no phases. Nothing to fit'.format(this=self)
            warnings.warn(msg, RuntimeWarning)
        # Step through each reflection in the relevant phases and find the peak
        for phase in self.phases:
            reflection_list += phase.reflection_list
            phase_peak_list = []
            for reflection in reflection_list:
                if contains_peak(scattering_lengths, reflection.qrange):
                    left = reflection.qrange[0]
                    right = reflection.qrange[1]
                    idx = np.where(np.logical_and(left < scattering_lengths,
                                                  scattering_lengths < right))
                    xs = scattering_lengths[idx]
                    ys = intensities[idx]
                    # Try each fit method until one works
                    for method in fitMethods:
                        newPeak = XRDPeak(reflection=reflection, method=method)
                        try:
                            newPeak.fit(x=xs, y=ys)
                        except exceptions.PeakFitError:
                            # Try next fit
                            continue
                        else:
                            phase_peak_list.append(newPeak)
                            break
                    else:
                        # No sucessful fit could be found.
                        msg = "RefinementWarning: peak could not be fit for {}.".format(reflection)
                        warnings.warn(msg, RuntimeWarning)
                assert False
            self._peak_list.append(phase_peak_list)
        return self._peak_list

    def predict(self, scattering_lengths):
        """Predict intensity values from the given scattering lengths, q.

        Arguments
        ---------
        - scattering_lengths : Iterable with scattering_lengths
          (x-values) that will be used to predict the diffraction
          intensities.
        """
        q = scattering_lengths
        predicted = np.zeros_like(q)
        # Calculate background
        if self.spline is not None:
            predicted += self.spline(q)
        # Calculate peak fittings
        for peak in self.peak_list:
            predicted += peak.predict(q)
        return predicted

    def plot(self, x, ax=None):
        warnings.warn(DeprecationWarning(), "Use predict() method and pyplot instead.")
        # Create new axes if necessary
        if ax is None:
            ax = plots.new_axes()
        # Check if background has been fit
        if self.spline is not None:
            # Plot background
            q = x
            background = self.spline(q)
            ax.plot(q, background)
        # Highlight peaks
        self.highlight_peaks(ax=ax)
        # Plot peak fittings
        peaks = []
        for peak in self.peak_list:
            # dataframes.append(peak.dataframe(background=spline))
            peaks.append(peak.predict())
            # peak.plot_overall_fit(ax=ax, background=spline)
        if peaks:
            predicted = pandas.concat(peaks)
            if self.spline:
                predicted = predicted + self.spline(predicted.index)
            predicted.plot(ax=ax)
        return ax

    def highlight_peaks(self, ax):
        color_list = [
            'green',
            'blue',
            'red',
            'orange'
        ]

        def draw_peaks(ax, phase, color):
            """Highlight the expected peak corresponding to this phase."""
            alpha = 0.15
            # Highlight each peak in this phase
            for reflection in phase.reflection_list:
                two_theta = reflection.qrange
                ax.axvspan(two_theta[0], two_theta[1],
                           color=color, alpha=alpha)
        # Highlight phases
        for idx, phase in enumerate(self.phases):
            draw_peaks(ax=ax, phase=phase, color=color_list[idx])
        # Highlight background
        for phase in self.background_phases:
            draw_peaks(ax=ax, phase=phase, color='grey')
        return ax

    def confidence(self):
        return 1
