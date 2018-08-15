# -*- coding: utf-8 -*-

import os

import numpy as np
import matplotlib.pyplot as plt

from . import plots, tube


class BaseRefinement():
    displacement_ratio = 0.
    zero_error = 0.
    lg_mix = 0.5
    # Status flags
    is_refined = {
        'displacement': False,
        'background': False,
        'peak_widths': False,
        'unit_cells': False,
        'scale_factors': False,
    }
    
    def __init__(self, phases=[], background_phases=[],
                 wavelengths=(), scan=None, num_bg_coeffs=5, file_root='temp_refinement'):
        """A base class for constructing refinements.
        
        Parameters
        ----------
        phases : optional
        background_phases : optional
        wavelengths : tuple, optional
          List of X-ray wavelengths, each entry is (wavelength,
          ratio), so copper K-alpha would be [(1.5406, 1), (1.5444,
          0.5)]
        num_bg_coeffs : int, optional
          How many terms to add to the ChebyShev polynomial.
        file_root : str, optional
          The base for any temporary files created during refinement.
        
        """
        self.phases = phases
        self.background_phases = background_phases
        self.num_bg_coeffs = num_bg_coeffs
        self.scan = scan
        self.wavelengths = wavelengths
        self.file_root = file_root
        # Reset all refinement status flags
        for key in self.is_refined.keys():
            self.is_refined[key] = False
    
    def predict(self, two_theta):
        """Return predicted diffraction intensities for given 2θ.
        
        Parameters
        ----------
        two_theta : np.ndarray
          Diffraction angles (2θ°) for predicting the diffraction
          pattern.
        
        Returns
        -------
        predicted : np.ndarray
          Diffraction intensities for values in ``two_theta``.
        
        """
        raise NotImplementedError
    
    def goodness_of_fit(self, two_theta=None, intensities=None):
        """Retrieve the degree of goodness for the refinement.
        
        The meaning of "goodness" depends on the specific type of
        refinement.
        
        Parameters
        ----------
        two_theta : np.ndarray, optional
          An array of 2θ° values.
        intensities : np.ndarray, optional
          Observed diffraction intensities for the 2θ values. Used for
          refinement if necessary.
        
        Returns
        -------
        goodness : np.ndarray
          A value describing how reliable the fit it.
        
        """
        raise NotImplementedError
    
    def background(self, two_theta=None, intensities=None):
        """Retrieve the predicted background array.
        
        If refinement of the background has not been done, this method
        will first perform the refinement process.
        
        Parameters
        ----------
        two_theta : np.ndarray, optional
          An array of 2θ° values.
        intensities : np.ndarray, optional
          Observed diffraction intensities for the 2θ values. Used for
          refinement if necessary.
        
        Returns
        -------
        background : np.ndarray
          The predicted background intensities for the 2θ values.
        
        """
        raise NotImplementedError
    
    def cell_params(self, two_theta, intensities):
        """Retrieve the predicted cell parameters.
        
        If refinement of the cell parameters has not been done, this
        method will first perform the refinement process.
        
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
          The predicted cell parameters
        
        """
        raise NotImplementedError
    
    def scale_factor(self, two_theta=None, intensities=None):
        """Retrieve the overall contribution for the whole pattern.
        
        If refinement of the scale_factor has not been done, this
        method will first perform the refinement process.
        
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
        raise NotImplementedError
    
    def broadenings(self, two_theta=None, intensities=None):
        """Retrieve the expected peak broadening for each phase.
        
        If refinement of the scale_factor has not been done, this
        method will first perform the refinement process. The exact
        meaning of "broadening" depends on the type of refinement
        used.
        
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
        raise NotImplementedError
    
    def phase_fractions(self, two_theta=None, intensities=None):
        """Retrieve the relative contribution from each phase.
        
        If refinement of the phase fractions has not been done, this
        method will first perform the refinement process.
        
        Parameters
        ----------
        two_theta : np.ndarray, optional
          An array of 2θ° values.
        intensities : np.ndarray, optional
          Observed diffraction intensities for the 2θ values. Used for
          refinement if necessary.
        
        Returns
        -------
        phase_fractions : tuple
          The predicted phase fractions, 1 for each phase.
        
        """
        raise NotImplementedError
    
    def peak_breadths(self, two_theta=None, intensities=None):
        raise NotImplementedError
    
    def plot(self, two_theta, intensities, ax=None):
        """Plot the data to compare the observations to the fit.
        
        The intensities vs two-theta will be plotted, along with the
        predicted fit, background and difference. If no axes is
        specified, a new one will be created.
        
        Parameters
        ----------
        two_theta : np.ndarray
          2θ° for the x-axis.
        intensities : np.ndarray
          Observed diffraction intensity values for the y-axis.
        ax : mpl.Axes, optional
          A matplotlib Axes() object to receive the plot. If omitted,
          a new one will be created.
        
        """
        if ax is None:
            ax = plots.new_axes()
            ax.xaxis.set_major_formatter(plots.DegreeFormatter())
        ax.plot(two_theta, intensities, linestyle="None", marker='+')
        # Plot the predicted pattern after refinement
        new_two_theta = np.linspace(np.min(two_theta), np.max(two_theta),
                                    num=10 * len(two_theta))
        background = self.background(new_two_theta)
        ax.plot(new_two_theta, background)
        predicted = self.predict(two_theta=new_two_theta)
        ax.plot(new_two_theta, predicted, linestyle=':')
        # Not plot the difference between the observed and actual
        diff = (intensities - self.predict(two_theta)) - np.max(intensities) / 20
        ax.plot(two_theta, diff, color="teal")
        ax.legend(['Actual', 'Background', 'Fit', 'Difference'])
