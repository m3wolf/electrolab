# -*- coding: utf-8 -*-

import os


class BaseRefinement():
    lg_mix = 0.5
    # Status flags
    is_refined = {
        'displacement': False,
        'background': False,
        'peak_widths': False,
        'unit_cells': False,
        'scale_factors': False,
    }
    
    def __init__(self, phases=[], background_phases=[], wavelengths=(), scan=None,
                 num_bg_coeffs=5):
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
    
    def refine_background(self):
        raise NotImplementedError
    
    def refine_unit_cells(self):
        raise NotImplementedError
    
    def scale_factors(self):
        raise NotImplementedError
    
    def plot(self):
        raise NotImplementedError
    
    def fwhm(self, phase=0):
        raise NotImplementedError
    
    def phase_weights(self):
        raise NotImplementedError
