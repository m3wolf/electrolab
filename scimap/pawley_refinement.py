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


def predict_diffractogram(two_theta, wavelengths, background_coeffs,
                          reflections, u=0, v=0, w=0, lg_mix=0.5):
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
    # Calculate the background function after normalizing
    x = (two_theta / 90) - 1
    y = np.zeros_like(x)
    bg = chebval(x, c=background_coeffs)
    y += bg
    # Add a peak for each reflection/wavelength combination
    for (wl, wl_ratio) in wavelengths:
        for (d, height, size, strain) in reflections:
            if 2*d < wl:
                warnings.warn('Invalid d-space {}Å for λ={}Å'
                              ''.format(d, wl), RuntimeWarning)
            else:
                # Calculate peak breadth
                theta = np.arcsin(wl/2/d)
                center = 2 * np.degrees(theta)
                beta_inst = np.sqrt(u*np.tan(theta)**2 + v * np.tan(theta) + w)
                K = 0.9 # Could be adjusted as needed...
                beta_size = K * wl / size / np.cos(theta)
                beta_strain = strain * np.tan(theta)
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
    
    def __init__(self, *args, **kwargs):
        # Create empty array of background coefficients
        super().__init__(*args, **kwargs)
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
          height, width).
        
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
    
    def refine(self, two_theta, intensities):
        """Perform a single refinement pass against certain parameters.
        
        Parameters
        ----------
        two_theta : np.ndarray
          An array of two_theta values, in degrees.
        intensities : np.ndarray
          Array with diffraction intensity at each two_theta value.
        
        """
        # Create a function to optimize against
        def residuals(params, two_theta, intensities):
            # First decode the parameters
            params = list(params)
            idx = 1 # For keeping track of list position
            # Make sure the lg_mix is between 0 and 1
            lg_mix = params[0]
            if lg_mix > 1 or lg_mix < 0:
                return np.zeros_like(two_theta, dtype='uint') - 1
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
                    # Prepare a tuple with the (pos, height, width)
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
                                        wavelengths=self.wavelengths,
                                        reflections=ref_list,
                                        lg_mix=0.5)
            return intensities - out
        # Prepare the error function to minimize
        errfunc = partial(residuals, two_theta=two_theta, intensities=intensities)
        # Perform the optimization
        p0 = [self.lg_mix] + list(self.bg_coeffs)
        p0 += (self.u, self.v, self.w)
        for phase in self.phases:
            p0 += phase.unit_cell.cell_parameters
            p0.append(phase.crystallite_size)
            p0.append(phase.strain)
            # Add height/width for each reflection
            refls = self.reflection_list(two_theta=two_theta, phases=[phase])
            for r in refls:
                p0.append(r[1])
            # for reflection in phase.reflection_list:
            #     p0.append(reflection.intensity)
            #     p0.append(reflection.fwhm)
        result = leastsq(errfunc, x0=p0)
        params, cov_x = result
        # Decode the parameters back into reflections, etc
        self.lg_mix = params[0]
        idx = 1
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
                # Prepare a tuple with the (pos, height, width)
                r.intensity = params[idx]
                idx += 1
    
    def predict(self, two_theta):
        """Predict diffraction intensity based on current parameters.
        
        Parameters
        ----------
        two_theta : np.ndarray
          2θ in degrees for calculating intensities.
        
        Returns
        -------
        out : np.ndarray
          Predicted diffraction intensities."""
        out = predict_diffractogram(two_theta,
                                    wavelengths=self.wavelengths,
                                    lg_mix=self.lg_mix,
                                    u=self.u, v=self.v, w=self.w,
                                    background_coeffs=self.bg_coeffs,
                                    reflections=self.reflection_list(two_theta=two_theta))
        return out
    
    def plot(self, two_theta, intensities, ax=None):
        if ax is None:
            ax = plots.new_axes()
            ax.xaxis.set_major_formatter(plots.DegreeFormatter())
        ax.plot(two_theta, intensities, linestyle="None", marker='+')
        # Plot the predicted pattern after refinement
        new_two_theta = np.linspace(np.min(two_theta), np.max(two_theta),
                                    num=10 * len(two_theta))
        predicted = self.predict(two_theta=new_two_theta)
        ax.plot(new_two_theta, predicted, linestyle=':')
        # Not plot the difference between the observed and actual
        
        diff = (intensities - self.predict(two_theta)) - np.max(intensities) / 20
        ax.plot(two_theta, diff, color="teal")
        ax.legend(['Actual', 'Fit', 'Difference'])
