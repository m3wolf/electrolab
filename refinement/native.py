# -*- coding: utf-8 -*-

import math
import warnings

import matplotlib.pyplot as plt
import numpy
import scipy
from scipy.interpolate import UnivariateSpline
import pandas

import exceptions
import plots
from xrd.peak import XRDPeak
from peakfitting import remove_peak_from_df
from refinement.base import BaseRefinement
from mapping.datadict import DataDict


class NativeRefinement(BaseRefinement):
    data_dict = DataDict(['spline', 'is_refined'])

    def peak_rms_error(self, phase, unit_cell=None):
        diffs = []
        wavelength = self.scan.wavelength
        predicted_peaks = phase.predicted_peak_positions(wavelength=wavelength,
                                                         unit_cell=unit_cell,
                                                         scan=self.scan)
        # Only include those that are within the two_theta range
        phase_idx = self.scan.phases.index(phase)
        actual_peaks = self._peak_list[phase_idx]
        # Make sure lists line up
        # if len(predicted_peaks) != len(actual_peaks):
        #     msg = 'uneven peak lists: {} != {}'.format(len(predicted_peaks),
        #                                                len(actual_peaks))
        #     raise exceptions.RefinementError(msg)
        # Prepare list of peak position differences
        for idx, actual_peak in enumerate(actual_peaks):
            offsets = [abs(p.two_theta-actual_peak.center_kalpha)
                       for p in predicted_peaks]
            diffs.append(min(offsets))
        # Calculate mean-square-difference
        running_total = 0
        for diff in diffs:
            running_total += diff**2
        rms_error = math.sqrt(running_total / len(diffs))
        return rms_error

    def refine_unit_cells(self, quiet=True):
        """Residual least squares refinement of the unit-cell
        parameters. Returns the residual root-mean-square error between
        predicted and observed 2θ."""
        # Fit peaks to Gaussian/Cauchy functions using least squares refinement
        self.fit_peaks()
        for phase in self.scan.phases:
            # Define an objective function that will be minimized
            def objective(cell_parameters):
                # Determine cell parameters from numpy array
                # Create a temporary unit cell and return the residual error
                unit_cell = phase.unit_cell.__class__()
                unit_cell.set_cell_parameters_from_list(cell_parameters)
                residuals = self.peak_rms_error(phase=phase,
                                                unit_cell=unit_cell)
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
                residual_error = self.peak_rms_error(phase=phase)
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

    def refine_background(self, scattering_lengths, intensities):
        # Remove pre-indexed peaks for background fitting
        phase_list = self.phases + self.background_phases
        df = pandas.Series(data=intensities, index=scattering_lengths)
        for phase in phase_list:
            for reflection in phase.reflection_list:
                remove_peak_from_df(reflection, df)
        # Determine a background line from the noise without peaks
        self.spline = UnivariateSpline(
            x=df.index,
            y=df.values,
            s=len(df) / 20,
            k=3
        )
        # Extrapolate the background for the whole pattern
        background = self.spline(scattering_lengths)
        # plt.figure(figsize=(9, 6))
        # plt.plot(scattering_lengths, background)
        # plt.plot(scattering_lengths, intensities)
        # plt.plot(scattering_lengths, intensities - background)
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
    def background(self):
        bg = getattr(self, '_background', None)
        if bg is None:
            self.refine_background()
            bg = self._background
        return bg

    @property
    def subtracted(self):
        subtracted = getattr(self, '_subtracted', None)
        if subtracted is None:
            self.refine_background()
            subtracted = self._subtracted
        return subtracted

    def refine_displacement(self):
        """Not implemented yet."""
        pass

    def net_area(self, two_theta_range):
        """Area under the scan diffractogram after background subtraction."""
        # fullDF = self.scan.diffractogram
        df = self.subtracted
        # Get peak dataframe for integration
        if self.scan.contains_peak(two_theta_range):
            netDF = df.loc[two_theta_range[0]:two_theta_range[1]]
            # Integrate peak
            area = numpy.trapz(x=netDF.index, y=netDF)
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

    def fit_peaks(self):
        """
        Use least squares refinement to fit gaussian/Cauchy/etc functions
        to the predicted reflections.
        """
        self._peak_list = []
        fitMethods = ['pseudo-voigt', 'gaussian', 'cauchy', 'estimated']
        reflection_list = []
        # Step through each reflection in the relevant phases and find the peak
        for phase in self.scan.phases:
            reflection_list += phase.reflection_list
            phase_peak_list = []
            for reflection in reflection_list:
                if self.scan.contains_peak(reflection.two_theta_range):
                    left = reflection.two_theta_range[0]
                    right = reflection.two_theta_range[1]
                    df = self.subtracted.loc[left:right]
                    # Try each fit method until one works
                    for method in fitMethods:
                        newPeak = XRDPeak(reflection=reflection, method=method)
                        try:
                            newPeak.fit(df)
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
            self._peak_list.append(phase_peak_list)
        return self._peak_list

    def plot(self, ax=None):
        # Check if background has been fit
        spline = getattr(self, 'spline', None)
        if spline is not None:
            # Create new axes if necessary
            if ax is None:
                ax = plots.new_axes()
            # Plot background
            two_theta = self.scan.diffractogram.index
            background = self.spline(two_theta)
            ax.plot(two_theta, background)
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
                two_theta = reflection.two_theta_range
                ax.axvspan(two_theta[0], two_theta[1],
                           color=color, alpha=alpha)
        # Highlight phases
        for idx, phase in enumerate(self.scan.phases):
            draw_peaks(ax=ax, phase=phase, color=color_list[idx])
        # Highlight background
        for phase in self.scan.background_phases:
            draw_peaks(ax=ax, phase=phase, color='grey')
        return ax

    def confidence(self):
        return 1
