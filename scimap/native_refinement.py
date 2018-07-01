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
# along with Scimap. If not, see <http://www.gnu.org/licenses/>.


import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
import scipy
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline, splrep, splev
from scipy.signal import savgol_filter
import pandas as pd

from . import exceptions
from . import plots
from .peak import XRDPeak
from .peakfitting import remove_peak_from_df
from .base_refinement import BaseRefinement


def contains_peak(scattering_lengths, qrange):
    """Does this instance have the given peak within its range of q values?"""
    qmax = max(scattering_lengths)
    qmin = min(scattering_lengths)
    isInRange = (qmin < qrange[0] < qmax or
                 qmin < qrange[1] < qmax)
    return isInRange


def peak_area(scattering_lengths, intensities, qrange):
    """Area under the scan diffractogram within the given range of
    scattering lengths (q)."""
    df = pd.Series(intensities, index=scattering_lengths)
    netDF = df.loc[qrange[0]:qrange[1]]
    # Integrate peak
    area = np.trapz(x=netDF.index, y=netDF)
    return area


class NativeRefinement(BaseRefinement):
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
    
    def unit_cells(self, scattering_lengths, intensities, quiet=True):
        """Residual least squares refinement of the unit-cell
        parameters. Returns an (p, 6) array where p is the number of
        phases and axis 1 has a value for each of the cell parameters
        (a, b, c, α, β, γ).
        """
        # Fit peaks to Gaussian/Cauchy functions using least squares refinement
        # self.fit_peaks(scattering_lengths=scattering_lengths,
        #                intensities=intensities)
        # Get a list of peak positions
        # assert len(self.phases) == 1 # Temporary to avoid weird fitting
        def peak_positions(scattering_lengths, intensities, phase):
            """Return a list of peak positions for the given phase."""
            peak_list = []
            for reflection in phase.reflection_list:
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
            return peak_list
        # Do a refinement for each phase
        residual_error = 0
        for phase in self.phases:
            peak_list = peak_positions(scattering_lengths, intensities, phase)
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
                residual_error += self.peak_rms_error(phase=phase, peak_list=peak_list)
            else:
                # Optimization failed for some reason
                raise exceptions.RefinementError(result.message)
        return residual_error
    
    def refine_phase_fractions(self, scattering_lengths, intensities):
        """Calculate the relative strengths of each phase in a diffractogram.
        
        The simplest approach is to calculate the peak area for each
        phases's diagnostic reflection. The fraction is then the ratio
        of each phases's reflection over the sum of all phases. This
        makes the assumption that the phases behave similarly and that
        the structure factor of the diagnostic reflections are also
        similar. By default this method does not remove the
        background.
        
        Parameters
        ----------
        scattering_lengths : np.ndarray
          Dependent variable for the diffractogram.
        intensities : np.ndarray
          Independent variable for the diffractogram. Must be the same
          shape as ``scattering_lengths``.
        
        Returns
        -------
        phase_fractions : np.ndarray
          The relative weight of each phase as determined by the
          diffraction pattern.
        
        """
        # Calculate the diagnistic peak area for each phase
        areas = []
        for p in self.phases:
            reflection = p.diagnostic_reflection
            area = peak_area(scattering_lengths, intensities, reflection.qrange)
            areas.append(area)
        # Divide by the total area to get relative phase fractions
        areas = np.array(areas)
        total_area = np.sum(areas)
        phase_fractions = areas / total_area
        return phase_fractions
    
    def refine_scale_factor(self, scattering_lengths, intensities):
        """Calculate the absolute strengths of each phase in a diffractogram.
        
        The simplest approach is to calculate the peak area for each
        phases's diagnostic reflection. The factor is then the sum of
        each phases's reflection over all phases. This makes the
        assumption that the phases behave similarly and that the
        structure factor of the diagnostic reflections are also
        similar. By default this method does not remove the
        background.
        
        Parameters
        ----------
        scattering_lengths : np.ndarray
          Dependent variable for the diffractogram.
        intensities : np.ndarray
          Independent variable for the diffractogram. Must be the same
          shape as ``scattering_lengths``.

        Returns
        -------
        phase_fractions : np.ndarray
          The relative weight of each phase as determined by the
          diffraction pattern.

        """
        # Calculate the diagnistic peak area for each phase
        areas = []
        for p in self.phases:
            reflection = p.diagnostic_reflection
            area = peak_area(scattering_lengths, intensities, reflection.qrange)
            areas.append(area)
        # Divide by the total area to get relative phase fractions
        areas = np.array(areas)
        total_area = np.sum(areas)
        return total_area
    
    def refine_background(self, scattering_lengths, intensities, s=None, k=4):
        
        """Fit a univariate spline to the background data.
        
        Arguments
        ---------
        - scattering_lengths : Array of scattering vector lengths, q.
        
        - intensities : Array of intensity values at each q position
        
        - s : Smoothing factor passed to the spline. Default is
            the variance of the background.
        
        - k : Degree of the spline (default quartic spline).
        """
        # Remove pre-indexed peaks for background fitting
        phase_list = self.phases + self.background_phases
        q, I = scattering_lengths, intensities
        for phase in phase_list:
            for reflection in phase.reflection_list:
                q, I = remove_peak_from_df(x=q, y=I, xrange=reflection.qrange)
        # Get an estimate for s from the non-peak data
        if s is None:
            s = np.std(I)
        # Smoothing for background fitting
        smoothI = savgol_filter(I, window_length=15, polyorder=5)
        # Determine a background line from the noise without peaks
        # self.spline = UnivariateSpline(
        #     x=q,
        #     y=I,
        #     s=s / 25,
        #     k=k,
        # )
        self.spline = Chebyshev(coef=np.ones((20,)))
        self.spline = self.spline.fit(q, smoothI, 10)
        # background = self.spline.cast(scattering_lengths)
        # self.spline = splrep(q, I)
        # Extrapolate the background for the whole pattern
        # background = self.spline(scattering_lengths)
        background = self.spline(scattering_lengths)
        return background
    
    def fwhm(self, phase_idx=0):
        """Use the previously fitted peaks to describe full-width and
        half-maximum."""
        raise NotImplementedError()
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
    
    def details(self):
        return "Native refinement"
    
    def peak_list_by_phase(self):
        """List of fitted peaks organized by phase."""
        peak_list = getattr(self, '_peak_list', None)
        if peak_list is None:
            msg = "Peak's not fit, please run {}.fit_peaks() first.".format(self)
            raise exceptions.RefinementError(msg)
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
        self._peak_list = []
        # fitMethods = ['pseudo-voigt', 'gaussian', 'cauchy', 'estimated']
        fitMethods = ['gaussian', 'cauchy']
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
                        msg = "RefinementWarning: peak could not be fit for {}."
                        msg = msg.format(reflection)
                        warnings.warn(msg, RuntimeWarning)
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
    
    # def plot(self, two_theta, intensities, ax=None):
        
    #     # Create new axes if necessary
    #     if ax is None:
    #         ax = plots.new_axes()
    #     # Check if background has been fit
    #     if self.spline is not None:
    #         # Plot background
    #         q = x
    #         background = self.spline(q)
    #         ax.plot(q, background)
    #     # Highlight peaks
    #     self.highlight_peaks(ax=ax)
    #     # Plot peak fittings
    #     peaks = []
    #     for peak in self.peak_list:
    #         # dataframes.append(peak.dataframe(background=spline))
    #         peaks.append(peak.predict())
    #         # peak.plot_overall_fit(ax=ax, background=spline)
    #     if peaks:
    #         predicted = pandas.concat(peaks)
    #         if self.spline:
    #             predicted = predicted + self.spline(predicted.index)
    #         predicted.plot(ax=ax)
    #     return ax
    
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
