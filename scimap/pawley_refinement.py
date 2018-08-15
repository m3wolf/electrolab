# -*- coding: utf-8 -*-
#
# Copyright © 2018 Mark Wolfman
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

import warnings
from functools import partial

import numpy as np
from numpy.polynomial.chebyshev import chebval
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

from .base_refinement import BaseRefinement
from . import exceptions
from . import plots


def size_broadening(two_theta, size, wavelength, K=0.9):
    """Calculate broadening due to crystallite size
    
    Parameters
    ----------
    two_theta : float
      2θ° scattering angle
    size : float
      Crystallite size in angstroms.
    wavelength : float
      X-ray wavelength in angstroms.
    K : float, optional
      Shape factor.
    
    Returns
    -------
    beta : float
      Integral breadth due to crystallite size.
    
    """
    theta = np.radians(two_theta / 2)
    wl = wavelength
    beta = K * wl / size / np.cos(theta)
    return beta


def strain_broadening(two_theta, strain):
    """Calculate peak broadening due to micro-strain.
    
    Parameters
    ----------
    two_theta : float
      2θ° scattering angle.
    strain : float
      Strain parameter.
    
    Returns
    -------
    beta : float
      Integral breadth due to crystallite size.
    
    """
    theta = np.radians(two_theta / 2)
    beta = strain * np.tan(theta)
    return beta


def instrument_broadening(two_theta, u, v, w):
    """Calculate the instrumental broadening at angle 2θ°."""
    theta = np.radians(two_theta / 2)
    beta_squared = u*np.tan(theta)**2 + v * np.tan(theta) + w
    if beta_squared > 0:
        beta = np.sqrt(beta_squared)
    else:
        beta = 0
    return beta


def gaussian(two_theta, center, height, breadth):
    """Produce a Gaussian probability distribution.
    
    Parameters
    ----------
    two_theta : np.ndarray
      Input values for the gaussian function (x-values).
    center : float
      Center position for the peak.
    height : float
      Scaling factor for the function.
    breadth : float
      Ratio of peak integral (area) over peak height.
    
    Returns
    -------
    out : np.ndarray
      Y values with the same shape as ``two_theta``.
    
    """
    x = two_theta
    x0 = center
    out = np.exp(-np.pi * np.square(x - x0) / np.square(breadth))
    return height * out


def cauchy(two_theta, center, height, breadth):
    """Produce a Cauchy (Lorentzian) probability distribution.
    
    Parameters
    ----------
    two_theta : np.ndarray
      Input values for the gaussian function (x-values).
    center : float
      Center position for the peak.
    height : float
      Scaling factor for the function.
    breadth : float
      Ratio of peak integral (area) over peak height.
    
    Returns
    -------
    out : np.ndarray
      Y values with the same shape as ``two_theta``.
    
    """
    w = breadth / np.pi
    x = two_theta
    x0 = center
    out = w**2 / (w**2 + (x-x0)**2)
    return height * out


def background(two_theta, coeffs):
    """Produce a chebyshev function using the given coefficients.
    
    Parameters
    ----------
    two_theta : np.ndarray
      2θ° to use as x-values. Will be normalized to the 0-180° range.
    coeffs : tuple
      Collection of coefficients to use for the Chebyshev pieces.
    
    Returns
    -------
    bg : np.ndarray
      Array with calculated background function of the same shape as
      ``two_theta``.
    
    """
    x = (two_theta / 90) - 1
    bg = chebval(x, c=coeffs)
    return bg


def predict_diffractogram(two_theta, wavelengths, background_coeffs,
                          reflections, displacement_ratio=0, u=0, v=0, w=0, lg_mix=0.5):
    """Take all the fitting parameters and compute a diffractogram.
    
    Parameters that are linked to a phase must all have that many
    entries in their first dimensions (eg. reflections,
    unit_cells). All lengths are in angstroms, and angles in degrees.
    
    Parameters
    ----------
    twotheta : np.ndarray
      Array with two theta values for which to compute diffraction
      intensity.
    wavelengths : tuple
      Tuple of wavelengths, in angstroms, of the X-ray radiation.
    background_coeffs : tuple
      Tuple of coefficients to build a Chebyshev polynomial
    displacement_ratio : float
      Distance of the sample from the instrument center of rotation,
      expressed as a ratio with the sample-to-detector distance.
    u, v, w : float
      Instrumental broadening parameters as described by the Coglioti
      equation.
    reflections : np.array
      A two-dimensional array with the shape of (reflection, param)
      where the params are (d, height, size, strain) with d in
      angstroms and FWHM in 2θ°.
    lg_mix : np.array
      Ratio of Cauchy (Lorentzian) to Gaussian shape. ``lg_mix=1``
      results in pure Cauchy and ``lg_mix=0`` results in pure
      Gaussian. Value is best when between 0 and 1.
    
    Returns
    -------
    intensities : np.ndarray
      The resulting diffraction pattern based on the given parameters.
    
    """
    y = np.zeros_like(two_theta)
    # Calculate the background function
    y += background(two_theta, coeffs=background_coeffs)
    # Add a peak for each reflection/wavelength combination
    for (wl, wl_ratio) in wavelengths:
        for (d, height, size, strain) in reflections:
            if 2*d < wl:
                warnings.warn('Invalid d-space {}Å for λ={}Å'
                              ''.format(d, wl), RuntimeWarning)
            else:
                center = 2 * np.degrees(np.arcsin(wl/2/d))
                # Adjust for displacement error
                center -= 2 * displacement_ratio * np.cos(center / 2)
                # Calculate peak breadth
                beta_inst = instrument_broadening(two_theta=center, u=u, v=v, w=w)
                beta_size = size_broadening(two_theta=center, size=size, wavelength=wl)
                beta_strain = strain_broadening(two_theta=center, strain=strain)
                # Add a gaussian curve
                beta_gauss = np.sqrt(beta_inst**2 + beta_size**2 + beta_strain**2)
                y += wl_ratio * lg_mix * cauchy(two_theta, center, height, beta_gauss)
                # Add a Cauchy curve
                beta_cauch = beta_inst + beta_size + beta_strain
                y += wl_ratio * (1-lg_mix) * gaussian(two_theta, center, height, beta_cauch)
    return y


class PawleyRefinement(BaseRefinement):
    u = 0
    v = 0
    w = 0
    
    def __init__(self, wavelengths, *args, **kwargs):
        # Create empty array of background coefficients
        super().__init__(wavelengths=wavelengths, *args, **kwargs)
        self.bg_coeffs = np.zeros(shape=(self.num_bg_coeffs,))
    
    def reflection_list(self, two_theta=(0, 180), phases=None, unit_cells=None, clip=False):
        """Return a list of reflections that are visible for this 2-theta range.
        
        Whether a reflection is valid is determined by the
        two_theta_range and the X-ray source wavelengths.
        
        Parameters
        ----------
        phases : list, optional
          A list of XRD phases to use. If omitted, the default
          ``self.phases`` will be used.
        unit_cells : list, optional
          A lit of cystal unit cells to use. If omitted, the unit cell
          present on each phase will be used. The length of this
          parameter should match the number of phases being used.
        clip : bool, optional
          If true, only reflections within the given two_theta range
          as provided.
        
        Returns
        -------
        reflections : list
          Each entry in ``reflections`` is a tuple of (d-space,
          height, crystallite_size, strain).
        
        """
        ref_list = []
        # Prepare default arguments
        if phases is None:
            phases = self.phases
        if unit_cells is None:
            unit_cells = [phase.unit_cell for phase in phases]
        # Determine min and max d-values for the given 2θ range
        wavelengths = [wl[0] for wl in self.wavelengths]
        d_min = np.min(wavelengths) / 2 / np.sin(np.radians(np.max(two_theta)/2))
        d_max = np.max(wavelengths) / 2 / np.sin(np.radians(np.min(two_theta)/2))
        # Prepare reflections for each phase
        for phase, unit_cell in zip(phases, unit_cells):
            for refl in phase.reflection_list:
                d_space = unit_cell.d_spacing(refl.hkl)
                if (d_min < d_space < d_max) or not clip:
                    reflection_params = (d_space, refl.intensity,
                                         phase.crystallite_size, phase.strain)
                    ref_list.append(reflection_params)
        return ref_list
    
    def scale_factor(self, two_theta, intensities=None):
        """Return the overall signal intensity for this refinement.
        
        This is approximated as the integrated area under the curve
        after background subtraction per unit angle 2θ.
        
        Parameters
        ----------
        two_theta : np.ndarray
          2θ° to use for calculating predicted curves and for running
          the refinement if not already done.
        intensities : np.ndarray, optional
          Used for running refinement if necessary.
        
        Returns
        -------
        scale_factor : float
          The approximate signal intensity for this refinement.
        
        """
        bg = self.background(two_theta=two_theta)
        Is = self.predict(two_theta=two_theta)
        two_theta_range = np.max(two_theta) - np.min(two_theta)
        area = np.trapz(Is-bg, x=two_theta)
        scale_factor = area / two_theta_range
        return scale_factor
    
    def phase_fractions(self, two_theta, intensities=None):
        """Return the relative weight of each phase in the refinement.
        
        Currently this is implemented by comparing the area under the
        predicted curve for each phase in ``self.phases`` after
        background subtraction. Since the individual peak intensities
        are refined for a Pawley refinement, the phase fraction cannot
        also be refined, so this implementation will only be
        meaningful if the phases are similar in the number and
        strength of their reflections.
        
        Parameters
        ----------
        two_theta : np.ndarray
          2θ° to use for calculating predicted curves and for running
          the refinement if not already done.
        intensities : np.ndarray, optional
          Used for running refinement if necessary.
        
        Returns
        -------
        fractions : tuple
          A tuple of phase fractions with length equal to the number
          of phases. The values will necessarily sum to 1.
        
        """
        bg = self.background(two_theta=two_theta)
        areas = []
        for phase in self.phases:
            # Get the integrated intensity for each phase
            Is = self.predict(two_theta=two_theta, phases=(phase,))
            area = np.trapz(Is-bg, x=two_theta)
            areas.append(area)
        # Convert to relative intensities
        total = sum(areas)
        fractions = tuple(a/total for a in areas)
        return fractions
    
    def peak_breadths(self, two_theta=90, intensities=None):
        """Retrieve the integral breadth for a peak in each phase with the
        given two_theta.
        
        The breadth at the given two-theta value is dependent on both
        instrument broadening and sample broadening. The instrument
        broadening is the same for all phases, whereas sample
        broadening will be different for each phase. Broadening from
        different sources is combined assuming Gaussian peak
        shapes. The mean wavelength used for refining will be used to
        calculate crystallite size broadening.
        
        Parameters
        ----------
        two_theta : int, np.ndarray, optional
          Two-theta values to use for calculating peak breadth (and
          optionally refining). Median value will be used for
          determining breadth.
        intensities : np.ndarray, optional
          Used for running refinement if necessary.
        
        Returns
        -------
        breadths : tuple
          An array with a peak breadth for each phase.
        
        """
        two_theta = np.median(two_theta)
        beta_inst = instrument_broadening(two_theta=two_theta,
                                          u=self.u, v=self.v, w=self.w)
        wavelength = np.mean(self.wavelengths)
        # Now produce broadening for each phase
        betas = []
        for phase in self.phases:
            beta_size = size_broadening(two_theta=two_theta,
                                        wavelength=wavelength, size=phase.crystallite_size)
            beta_strain = strain_broadening(two_theta=two_theta, strain=phase.strain)
            betas.append(np.sqrt(beta_inst**2 + beta_size**2 + beta_strain**2))
        return tuple(betas)
    
    def unit_cells(self, two_theta=None, intensities=None):
        """Retrieve unit-cell parameters for each phase.
        
        Parameters
        ----------
        two_theta : np.ndarray, optional
          Used for refining, if necessary
        intensities : np.ndarray, optional
          Used for refining, if necessary
        
        Returns
        -------
        cell_params : tuple
          (a, b, c, α, β, γ) for each phase present, so for 2 phases,
          12 values are returned.
        
        """
        cell_params = []
        for phase in self.phases:
            cell_params.extend(phase.unit_cell.as_tuple())
        return cell_params
    
    def background(self, two_theta, intensities=None):
        """Retrieve the fitted background for the scan.
        
        Parameters
        ----------
        two_theta : np.ndarray
          2θ° to use as x-values
        intensities : np.ndarray, optional
          Used for refining, if necessary
        
        """
        bg = background(two_theta, coeffs=self.bg_coeffs)
        return bg
    
    def refine(self, two_theta, intensities):
        """Perform a single refinement pass against certain parameters.
        
        Parameters
        ----------
        two_theta : np.ndarray
          An array of 2θ° to use as x-values.
        intensities : np.ndarray
          Array with diffraction intensity at each two_theta value.
        
        """
        # Create a function to optimize against
        def residuals(params, two_theta, intensities):
            # First decode the parameters
            params = list(params)
            idx = 0 # For keeping track of list position
            displacement_ratio = params[0]
            # Make sure the lg_mix is between 0 and 1
            lg_mix = params[1]
            if lg_mix > 1 or lg_mix < 0:
                return np.zeros_like(two_theta, dtype='uint') - 1
            idx = 2 # lg_mix and sample displacement have been read
            # Extract background coefficients
            bg_coeffs = params[idx:idx+self.num_bg_coeffs]
            idx += self.num_bg_coeffs
            # Extract u, v and w peak breadth parameters
            u, v, w = params[idx:idx+3]
            idx += 3
            # Prepare a list of all reflections
            ref_list = []
            # Prepare unit cell and reflection list
            for phase in self.phases:
                # Unit cell
                PUnitCell = phase.unit_cell.__class__
                unit_cell = PUnitCell()
                num_params = len(unit_cell.free_parameters)
                cell_params = params[idx:idx+num_params]
                unit_cell.set_cell_parameters_from_list(cell_params)
                idx += num_params
                phase.crystallite_size = params[idx]
                phase.strain = params[idx+1]
                idx += 2
                # Reflection list
                refls = self.reflection_list(two_theta=two_theta,
                                             phases=[phase],
                                             unit_cells=[unit_cell])
                for r in refls:
                    # Punish negative peak heights
                    if params[idx] < 0:
                        return np.full_like(two_theta, fill_value=np.max(intensities))
                    else:
                        # Prepare a tuple with the (d, height, size, strain)
                        ref = (
                            r[0],
                            params[idx],
                            r[2],
                            r[3],
                        )
                        ref_list.append(ref)
                        idx += 1
            # Check that there are no extra parameters
            if idx != len(params):
                raise exceptions.RefinementError('Extra parameters: '
                                                 '{} given, {} used.'
                                                 ''.format(len(params), idx))
            # Calculate difference from observed pattern
            out = predict_diffractogram(two_theta=two_theta,
                                        background_coeffs=bg_coeffs,
                                        displacement_ratio=displacement_ratio,
                                        wavelengths=self.wavelengths,
                                        reflections=ref_list,
                                        u=u, v=v, w=w,
                                        lg_mix=0.5)
            return intensities - out
        # Prepare the error function to minimize
        errfunc = partial(residuals, two_theta=two_theta, intensities=intensities)
        # Perform the optimization
        p0 = [self.displacement_ratio] + [self.lg_mix] + list(self.bg_coeffs)
        p0 += (self.u, self.v, self.w)
        for phase in self.phases:
            p0 += phase.unit_cell.cell_parameters
            p0.append(phase.crystallite_size)
            p0.append(phase.strain)
            # Add height/width for each reflection
            refls = self.reflection_list(two_theta=two_theta, phases=[phase])
            for r in refls:
                p0.append(r[1])
        result = leastsq(errfunc, x0=p0, maxfev=len(p0)*500)
        params, cov_x = result
        # Decode the parameters back into reflections, etc
        self.displacement_ratio = params[0]
        self.lg_mix = params[1]
        idx = 2
        self.bg_coeffs = params[idx:idx + self.num_bg_coeffs]
        idx += self.num_bg_coeffs
        self.u = params[idx]
        self.v = params[idx+1]
        self.w = params[idx+2]
        idx += 3
        # Decode unit cells
        for phase in self.phases:
            # Unit cell
            num_params = len(phase.unit_cell.free_parameters)
            cell_params = params[idx:idx+num_params]
            idx += num_params
            phase.unit_cell.set_cell_parameters_from_list(cell_params)
            phase.crystallite_size = params[idx]
            phase.strain = params[idx+1]
            idx += 2
            # Reflection list
            for r in phase.reflection_list:
                r.intensity = params[idx]
                idx += 1
    
    def predict(self, two_theta, phases=None):
        """Predict diffraction intensity based on current parameters.
        
        Parameters
        ----------
        two_theta : np.ndarray
          2θ in degrees for calculating intensities.
        phases : list, optional
          A list of XRD phases to use. If omitted, the default
          ``self.phases`` will be used.
        
        Returns
        -------
        out : np.ndarray
          Predicted diffraction intensities."""
        reflections = self.reflection_list(two_theta=two_theta, phases=phases)
        out = predict_diffractogram(two_theta,
                                    wavelengths=self.wavelengths,
                                    lg_mix=self.lg_mix,
                                    u=self.u, v=self.v, w=self.w,
                                    background_coeffs=self.bg_coeffs,
                                    reflections=reflections)
        return out
