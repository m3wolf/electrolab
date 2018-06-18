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
                 wavelengths=(), scan=None, num_bg_coeffs=5):
        """A base class for constructing refinements.
        
        Parameters
        ----------
        phases : optional
        background_phases : optional
        wavelengths : tuple
          List of X-ray wavelengths, each entry is (wavelength,
          ratio), so copper K-alpha would be [(1.5406, 1), (1.5444,
          0.5)]
        
        """
        self.phases = phases
        self.background_phases = background_phases
        self.num_bg_coeffs = num_bg_coeffs
        self.scan = scan
        self.wavelengths = wavelengths
    
    def predict(self, two_theta):
        raise NotImplementedError
    
    def background(self, two_theta=None, intensities=None):
        raise NotImplementedError
    
    def unit_cells(self):
        raise NotImplementedError
    
    def scale_factor(self, two_theta=None, intensities=None):
        raise NotImplementedError
    
    def fwhm(self, phase=0):
        raise NotImplementedError
    
    def phase_fractions(self, two_theta=None, intensities=None):
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
