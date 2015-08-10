# -*- coding: utf-8 -*-

from collections import namedtuple
import math

import scipy

import exceptions
from xrd.peak import XRDPeak
from xrd.reflection import hkl_to_tuple

class Phase():
    """A crystallographic phase that can be found in a Material."""
    reflection_list = [] # Predicted peaks by crystallography
    spacegroup = '' # To be subclassed
    scale_factor = 1 # Contribution to the overall pattern in refinement
    eta = 0.5 # Pseudo-voigt peak-shape balance between gaussian/cauchy fit

    def __repr__(self):
        name = getattr(self, 'name', '[blank]')
        return "<{}: {}>".format(self.__class__.__name__, name)

    def reflection_by_hkl(self, hkl_input):
        for reflection in self.reflection_list:
            if reflection.hkl == hkl_to_tuple(hkl_input):
                return reflection

    @property
    def diagnostic_reflection(self):
        reflection = self.reflection_by_hkl(self.diagnostic_hkl)
        return reflection

    @diagnostic_reflection.setter
    def diagnostic_reflection(self, new_hkl):
        self.diagnostic_hkl = new_hkl

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
            raise exceptions.RefinementError(result.message)

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
            # Peak fitting has not been performed, raise an error
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
                newPeak = XRDPeak(reflection=reflection)
                df = scan.diffractogram.loc[
                    reflection.two_theta_range[0]:reflection.two_theta_range[1]
                ]
                # Try each fit method until one works
                for method in fitMethods:
                    try:
                        newPeak.fit(df.index, df.subtracted, method=method)
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

    def highlight_peaks(self, ax, color='green'):
        """Highlight the expected peak corresponding to this phase."""
        alpha = 0.15
        # Highlight each peak in this phase
        for reflection in self.reflection_list:
            two_theta = reflection.two_theta_range
            ax.axvspan(two_theta[0], two_theta[1], color=color, alpha=alpha)
        return ax

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
