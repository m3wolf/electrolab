# -*- coding: utf-8 -*-
"""Sample definitions and refinements for lithium manganese oxide
LiMn_2O_4"""

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit, minimize

from .xrd_map import XRDMap
from .phase import Phase
from .standards import Aluminum
from .unitcell import CubicUnitCell, TetragonalUnitCell
from .reflection import Reflection
from .base_refinement import BaseRefinement
from .fullprof_refinement import FullProfPhase, FullprofRefinement
from .utilities import twotheta_to_q


class CubicLMO(Phase):
    name = 'cubic LiMn2O4'
    unit_cell = CubicUnitCell(a=8.13)
    spacegroup = 'Fd-3m'
    fullprof_spacegroup = 'F D -3 M'
    diagnostic_hkl = '333'
    reflection_list = [
        Reflection('111', multiplicity=8, intensity=7.344,
                   qrange=(1.24, 1.38)),
        Reflection('220', multiplicity=12, intensity=0.036,
                   qrange=(2.11, 2.18)),
        Reflection('311', multiplicity=12, intensity=3.908,
                   qrange=(2.48, 2.59)),
        Reflection('222', multiplicity=8, intensity=1.024,
                   qrange=(2.61, 2.65)),
        Reflection('400', multiplicity=6, intensity=5.228,
                   qrange=(3.01, 3.12)),
        Reflection('331', multiplicity=24, intensity=1.301,
                   qrange=(3.32, 3.38)),
        Reflection('422', multiplicity=24, intensity=0.428,
                   qrange=(3.70, 3.76)),
        Reflection('511', multiplicity=8, intensity=2.111,
                   qrange=(3.89, 4.08)),
        Reflection('333', multiplicity=8, intensity=0.007,
                   qrange=(3.89, 4.08)),
        Reflection('440', multiplicity=12, intensity=3.162,
                   qrange=(4.29, 4.44)),
        Reflection('531', multiplicity=48, intensity=1.406,
                   qrange=(4.29, 4.44)),
        Reflection('442', multiplicity=24, intensity=0.070,
                   qrange=(4.50, 4.62)),
        # Reflection('620', multiplicity=24, intensity=0.031,
        #            qrange=(4.83, 4.91)),
        # Reflection('533', multiplicity=24, intensity=0.449,
        #            qrange=(4.91, 4.99)),
        # Reflection('622', multiplicity=24, intensity=0.185,
        #            qrange=(4.99, 5.07)),
    ]


class TetragonalLMO(Phase):
    unit_cell = TetragonalUnitCell()
    diagnostic_hkl = None
    reflection_list = [
        Reflection('000', qrange=(2.75, 2.82)),
    ]


class HighVPhase(FullProfPhase, CubicLMO):
    unit_cell = CubicUnitCell(a=8.053382)
    diagnostic_hkl = '333'
    reflection_list = [
        Reflection('333', qrange=(4.03, 4.08), multiplicity=8, intensity=0.081),
        Reflection('511', qrange=(4.03, 4.08), multiplicity=24, intensity=23.207),
        Reflection('440', qrange=(4.38, 4.44), multiplicity=12, intensity=39.180),
        Reflection('531', qrange=(4.59, 4.63), multiplicity=48, intensity=14.616),
    ]
    u = -0.139195
    v = 0.198405
    w = 0.008828
    I_g = -0.033578
    eta = 0.386
    x = 0.013
    isotropic_temp = -2.1903
    scale_factor = 0.05


class MidVPhase(FullProfPhase, CubicLMO):
    unit_cell = CubicUnitCell(a=8.12888)
    diagnostic_hkl = '333'
    reflection_list = [
        Reflection('333', qrange=(3.98, 4.03), multiplicity=8, intensity=0.081),
        Reflection('511', qrange=(3.98, 4.03), multiplicity=24, intensity=23.207),
        Reflection('440', qrange=(4.33, 4.38), multiplicity=12, intensity=39.180),
        Reflection('531', qrange=(4.54, 4.58), multiplicity=48, intensity=14.616),
    ]
    scale_factor = 0.05
    u = -0.139195
    v = 0.198405
    w = 0.033169
    I_g = 0
    eta = 0.209160
    x = 0.013
    isotropic_temp = -3.7293


# Prepare materials with new reflections
class MidV440Phase(MidVPhase):
    diagnostic_hkl = '440'


class HighV440Phase(HighVPhase):
    diagnostic_hkl = '440'


class MidV531Phase(MidVPhase):
    diagnostic_hkl = '531'


class HighV531Phase(HighVPhase):
    diagnostic_hkl = '531'


class LMORefinement(FullprofRefinement):
    bg_coeffs = [0.409, 14.808, -14.732, -10.292, 34.249, -28.046]
    zero = -0.001360
    displacement = 0.000330
    transparency = -0.008100


# Define a new class for mapping the two-phase plateau
class LMOPlateauMap(XRDMap):
    scan_time = 300
    two_theta_range = (55, 70)
    phases = [HighVPhase, MidVPhase]
    background_phases = [Aluminum]
    phase_ratio_normalizer = Normalize(0, 0.7, clip=True)
    reliability_normalizer = Normalize(0.7, 2, clip=True)


class TwoPhaseRefinement(BaseRefinement):
    peak_ranges = [
        (58.0, 60.5),
        (64.0, 66.3),
        (67.3, 69.6),
    ]
    approx_peak_positions = [58.9, 59.65, 64.75, 65.5, 68.1, 68.8]
    def remove_peaks(self, two_theta, intensities):
        """Remove peaks from the dataset, leaving only the background.
        
        Returns
        =======
        two_theta
          Two theta array with only non-peak values.
        intensities
          Intensity array with only non-peak values.
        
        """
        # Determine where the peaks are
        is_peak = np.zeros_like(two_theta, dtype=bool)
        for peak in self.peak_ranges:
            is_peak = np.logical_or(
                is_peak,
                np.logical_and(two_theta>peak[0], two_theta<peak[1])
            )
        # Remove peaks from the data
        red_two_theta = two_theta[~is_peak]
        red_intensities = intensities[~is_peak]
        return red_two_theta, red_intensities
    
    def background(self, two_theta, intensities):
        spline = Chebyshev(coef=np.ones((20,)))
        red_two_theta, red_intensities = self.remove_peaks(
            two_theta, intensities)
        # Fit the background using the reduced dataset
        spline = spline.fit(red_two_theta, red_intensities, 10)
        # Return the predicted background
        background = np.array(spline(two_theta))
        return background
    
    def peaks(self, two_theta, *params):
        params = np.array(params).reshape(-1, 3)
        predicted = np.zeros_like(two_theta)
        for (center, height, width) in params:
            c = width / 2.35482
            b = center
            a = height
            predicted += a * np.exp(-((two_theta-b)**2)/(2*c**2))
        return predicted
    
    def fit_peaks(self, two_theta, intensities):
        bg = self.background(two_theta, intensities)
        subtracted = intensities - bg
        # Find starting parameters for peak centers and heights
        approx_centers = self.approx_peak_positions
        approx_heights = [0] * len(approx_centers)
        approx_widths = [0.25] * len(approx_centers)
        for i in range(len(approx_centers)):
            center = approx_centers[i]
            is_peak = np.logical_and(two_theta > center-0.3, two_theta < center+0.3)
            real_center = two_theta[is_peak][np.argmax(subtracted[is_peak])]
            approx_centers[i] = real_center
            height = np.max(subtracted[is_peak])
            approx_heights[i] = height
        # Construct the initial guess (p0)
        p0 = np.array(list(zip(approx_centers, approx_heights, approx_widths)))
        p0 = p0.ravel()
        # Do the fitting
        popt, pcov = curve_fit(self.peaks, two_theta, subtracted, p0=p0)
        popt = popt.reshape(-1, 3)
        return popt
    
    def peak_areas(self, two_theta, intensities):
        """Integrated area for each peak in the pattern.
        
        Background is subtracted before calculating area.
        
        Parameters
        ----------
        two_theta : np.ndarray, optional
          An array of 2θ° values.
        intensities : np.ndarray, optional
          Observed diffraction intensities for the 2θ values. Used for
          refinement if necessary.
        
        Returns
        -------
        areas : tuple
          The integrated area for each peak in the pattern.
        
        """
        # Calculate peak areas
        peaks = self.fit_peaks(two_theta, intensities)
        areas = [(p[1] * p[2] / 0.3989) for p in peaks]
        return areas
    
    def peak_heights(self, two_theta, intensities):
        """Height of maximum intensity for each peak in the pattern.
        
        Background is subtracted before calculating height.
        
        Parameters
        ----------
        two_theta : np.ndarray, optional
          An array of 2θ° values.
        intensities : np.ndarray, optional
          Observed diffraction intensities for the 2θ values. Used for
          refinement if necessary.
        
        Returns
        -------
        areas : tuple
          The height of each peak in the pattern.
        
        """
        peaks = self.fit_peaks(two_theta, intensities)
        heights = np.array([p[1] for p in peaks])
        return heights
    
    def peak_breadths(self, two_theta, intensities):
        """Breadth for each peak in the pattern.
        
        After background subtraction, the integral breadth is the area
        under a peak divided by its height.
        
        Parameters
        ----------
        two_theta : np.ndarray, optional
          An array of 2θ° values.
        intensities : np.ndarray, optional
          Observed diffraction intensities for the 2θ values. Used for
          refinement if necessary.
        
        Returns
        -------
        areas : tuple
          The breadth of each peak in the pattern.
        
        """
        areas = self.peak_areas(two_theta, intensities)
        heights = self.peak_heights(two_theta, intensities)
        breadths = areas / heights
        return breadths
    
    def phase_fractions(self, two_theta, intensities):
        areas = self.peak_areas(two_theta, intensities)
        # Find phase fractions based on total peak areas
        w0 = np.sum(areas[0::2])
        w1 = np.sum(areas[1::2])
        fractions = np.array([w0, w1]) / np.sum([w0, w1])
        return fractions
    
    def predict(self, two_theta, intensities):
        peaks = self.fit_peaks(two_theta, intensities)
        background = self.background(two_theta, intensities)
        predicted = background + self.peaks(two_theta, *peaks)
        return np.array(predicted)
    
    def goodness_of_fit(self, two_theta, intensities):
        """Retrieve the degree of goodness for the refinement.
        
        This returns a value for the RMS residual error divided by the
        RMS for the original signal, one a scale from 0 to 1 where 1
        means there is no residual error, and 0 means the error is
        similar to the signal.
        
        Parameters
        ----------
        two_theta : np.ndarray, optional
          An array of 2θ° values.
        intensities : np.ndarray, optional
          Observed diffraction intensities for the 2θ values.
        
        Returns
        -------
        goodness : np.ndarray
          A value describing how reliable the fit it.
        
        """
        predicted = self.predict(two_theta, intensities)
        rms_error = np.sqrt(np.mean(np.power(intensities - predicted, 2)))
        rms_signal = np.sqrt(np.mean(np.power(intensities - np.min(intensities), 2)))
        goodness = 1 - (rms_error / rms_signal)
        return goodness
    
    def cell_params(self, two_theta, intensities):
        """Retrieve the predicted cell parameters.
        
        Parameters
        ----------
        two_theta : np.ndarray, optional
          An array of 2θ° values.
        intensities : np.ndarray, optional
          Observed diffraction intensities for the 2θ values. Used for
          refinement if necessary.
        
        Returns
        -------
        cell_params : tuple
          The predicted cell parameters for each phase, as tuples of
          (a, b, c, α, β, γ) values.
        
        """
        # Get list of reflection positions
        peaks = self.fit_peaks(two_theta, intensities)[:,0]
        cell_params = [
            self._refine_unit_cell(peaks[0::2], (4.2, 4.2, 4.2, 90, 90, 120)),
            self._refine_unit_cell(peaks[1::2], (4.2, 4.2, 4.2, 90, 90, 120)),
        ]
        return cell_params
    
    def _refine_unit_cell(self, peak_positions, p0):
        """Refine the unit cell parameter for a given set of reflections."""
        hkls = ('333', '440', '531')
        wl = np.sum([w[0] * w[1] for w in self.wavelengths]) / np.sum([w[1] for w in self.wavelengths])
        unit_cell = CubicUnitCell()
        # Define objective minimization function
        def objective(params):
            # Update the unit cell with requrested parameters
            unit_cell.set_cell_parameters_from_list(params)
            # Calculate reflections for hkl values
            d = np.array([unit_cell.d_spacing(hkl) for hkl in hkls])
            q_calc = 2 * np.pi / d
            # Compare predicted reflection positions to observed
            q_obs = twotheta_to_q(peak_positions, wavelength=wl)
            rms_error = np.sqrt(np.mean(np.power(q_calc - q_obs, 2)))
            return rms_error
        # Perform optimization to get cell parameters (+ displacement)
        res = minimize(objective, tuple(p0))
        # Calculate true cell parameters
        unit_cell.set_cell_parameters_from_list(res.x)
        cell_params = unit_cell.all_parameters
        # Calculate predicted reflection positions
        # Not used for now, but easy to print for troubleshooting
        d = np.array([unit_cell.d_spacing(hkl) for hkl in hkls])
        twotheta = 2 * np.degrees(np.arcsin(wl / 2 / d))
        return cell_params
    
    def scale_factor(self, two_theta, intensities):
        """Retrieve the overall contribution for the whole pattern.
       
        Parameters
        ----------
        two_theta : np.ndarray, optional
          An array of 2θ° values.
        intensities : np.ndarray, optional
          Observed diffraction intensities for the 2θ values. Used for
          refinement if necessary.
        
        Returns
        -------
        scale_factor : float
          The predicted scale factor for the pattern.
        
        """
        areas = self.peak_areas(two_theta, intensities)
        area = np.sum(areas)
        return area
    
    def broadenings(self, two_theta=None, intensities=None):
        """Retrieve the expected peak broadening for each phase.
        
        Parameters
        ----------
        two_theta : np.ndarray, optional
          An array of 2θ° values.
        intensities : np.ndarray, optional
          Observed diffraction intensities for the 2θ values. Used for
          refinement if necessary.
        
        Returns
        -------
        broadenings : tuple
          The predicted peak broadenings for each phase.
        
        """
        # Calculate broadenings for each peak
        breadths = self.peak_breadths(two_theta, intensities)
        # Reduce all peaks to two phases
        broadenings = [np.mean(breadths[::2]), np.mean(breadths[1::2])]
        return broadenings


class SolidSolutionRefinement(TwoPhaseRefinement):
    approx_peak_positions = [58.8, 64.5, 67.9]
    
    def phase_fractions(self, two_theta, intensities):
        """This is a 1-phase system, so the phase ratio is always 1."""
        return [1.]
    
    def cell_params(self, two_theta, intensities):
        """Retrieve the predicted cell parameters.
        
        Parameters
        ----------
        two_theta : np.ndarray, optional
          An array of 2θ° values.
        intensities : np.ndarray, optional
          Observed diffraction intensities for the 2θ values. Used for
          refinement if necessary.
        
        Returns
        -------
        cell_params : tuple
          The predicted cell parameters for each phase, as tuples of
          (a, b, c, α, β, γ) values.
        
        """
        # Get list of reflection positions
        peaks = self.fit_peaks(two_theta, intensities)[:,0]
        cell_params = [
            self._refine_unit_cell(peaks, (4.2, 4.2, 4.2, 90, 90, 120)),
        ]
        return cell_params
    
    def broadenings(self, two_theta=None, intensities=None):
        """Retrieve the expected peak broadening for each phase.
        
        Parameters
        ----------
        two_theta : np.ndarray, optional
          An array of 2θ° values.
        intensities : np.ndarray, optional
          Observed diffraction intensities for the 2θ values. Used for
          refinement if necessary.
        
        Returns
        -------
        broadenings : tuple
          The predicted peak broadenings for each phase.
        
        """
        # Calculate broadenings for each peak
        breadths = self.peak_breadths(two_theta, intensities)
        # Reduce all peaks to two phases
        broadenings = [np.mean(breadths)]
        return broadenings
