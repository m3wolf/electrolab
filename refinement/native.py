# -*- coding: utf-8 -*-

import math
import numpy
import scipy
from scipy.interpolate import UnivariateSpline
import pandas

import exceptions
import plots
from xrd.peak import remove_peak_from_df, XRDPeak
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
        df = self.scan
        phase_idx = self.scan.phases.index(phase)
        actual_peaks = self._peak_list[phase_idx]
        # Make sure lists line up
        if len(predicted_peaks) != len(actual_peaks):
            msg = 'uneven peak lists: {} != {}'.format(len(predicted_peaks),
                                                       len(actual_peaks))
            raise exceptions.RefinementError(msg)
        # Prepare list of peak position differences
        for idx, actual_peak in enumerate(actual_peaks):
            actual = actual_peak.center_kalpha
            predicted = predicted_peaks[idx].two_theta
            diffs.append(actual-predicted)
        # Calculate mean-square-difference
        running_total = 0
        for diff in diffs:
            running_total += diff**2
        rms_error = math.sqrt(running_total/len(diffs))
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
                phase.unit_cell.set_cell_parameters_from_list(optimized_parameters)
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

    def refine_background(self):
        originalData = self.scan.diffractogram
        workingData = originalData.copy()
        # Remove pre-indexed peaks for background fitting
        phase_list = self.scan.phases + self.scan.background_phases
        for phase in phase_list:
            for reflection in phase.reflection_list:
                remove_peak_from_df(reflection, workingData)
        # Determine a background line from the noise without peaks
        self.spline = UnivariateSpline(
            x=workingData.index,
            y=workingData.counts,
            s=len(workingData.index)*25,
            k=4
        )
        # Extrapolate the background for the whole pattern
        x = originalData.index
        self._background = pandas.Series(data=self.spline(x),
                                         index=originalData.index)
        self._subtracted = pandas.Series(originalData.counts - self._background)
        return originalData

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

    @property
    def peak_list(self):
        peak_list = getattr(self, '_peak_list', None)
        if peak_list is None:
            peak_list = self.fit_peaks()
        # Flatten the list of phases/peaks
        full_list = []
        for phase in peak_list:
            full_list += phase
        return full_list

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
                    newPeak = XRDPeak(reflection=reflection)
                    df = self.subtracted.loc[
                        reflection.two_theta_range[0]:reflection.two_theta_range[1]
                    ]
                    # Try each fit method until one works
                    for method in fitMethods:
                        try:
                            newPeak.fit(two_theta=df.index,
                                        intensity=df,
                                        method=method)
                        except exceptions.PeakFitError:
                            # Try next fit
                            continue
                        else:
                            phase_peak_list.append(newPeak)
                            break
                    else:
                        # No sucessful fit could be found.
                        msg = "peak could not be fit for {}.".format(reflection)
                        print(msg)
            self._peak_list.append(phase_peak_list)
        return self._peak_list

    def plot(self, ax=None):
        # Check if background has been fit
        spline = getattr(self, 'spline', None)
        if spline is not None:
            # Create new axes if necessary
            if ax is None:
                ax=plots.new_axes()
            # Plot background
            two_theta = self.scan.diffractogram.index
            background = self.spline(two_theta)
            ax.plot(two_theta, background)
        # Highlight peaks
        self.highlight_peaks(ax=ax)
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
                ax.axvspan(two_theta[0], two_theta[1], color=color, alpha=alpha)
        # Highlight phases
        for idx, phase in enumerate(self.scan.phases):
            draw_peaks(ax=ax, phase=phase, color=color_list[idx])
        # Highlight background
        for phase in self.scan.background_phases:
            draw_peaks(ax=ax, phase=phase, color='grey')
        return ax
