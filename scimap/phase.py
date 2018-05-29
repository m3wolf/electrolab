# -*- coding: utf-8 -*-

import warnings
from collections import namedtuple
import copy
import math

from .reflection import hkl_to_tuple
from .unitcell import UnitCell


class Phase():
    """A crystallographic phase that can be found in a Material."""
    name = None
    reflection_list = []  # Predicted peaks by crystallography
    spacegroup = ''
    scale_factor = 1
    unit_cell = UnitCell
    # data_dict = PhaseDataDict(['scale_factor', 'u', 'v', 'w'])
    # Profile peak-width parameters (fwhm = u*(tan θ)^2 + v*tan θ + w)
    u = 0
    v = 0
    w = 0

    def __init__(self):
        # Create a fresh unit cell
        self.unit_cell = copy.copy(self.unit_cell)

    def __str__(self):
        name = self.name
        if name is None:
            name = 'generic phase'
        return name

    def __repr__(self):
        name = self.name
        if name is None:
            name = '[blank]'
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

    def predicted_peak_positions(self, *args, **kwargs):
        warnings.warn("Use ``predicted_peaks()`` instead", DeprecationWarning)
        return self.predicted_peaks(*args, **kwargs)
    
    def predicted_peaks(self, unit_cell=None, scan=None):
        """Use the space group of this phase's unit cell to predict where the
        peaks will be."""
        # Use current unit_cell if none is given
        if unit_cell is None:
            unit_cell = self.unit_cell
        PredictedPeak = namedtuple('PredictedPeak', ('hkl', 'd', 'q'))
        predicted_peaks = []
        for reflection in self.reflection_list:
            # Only include reflection if it's within the scan's two-theta range
            if scan is not None:
                if not scan.contains_peak(reflection.two_theta_range):
                    continue
            # Calculate predicted position
            hkl = reflection.hkl
            d = unit_cell.d_spacing(hkl)
            # wavelength = 1.5418
            # radians = math.asin(wavelength / 2 / d)
            # twotheta = 2 * math.degrees(radians)
            q = 2 * math.pi / d
            predicted_peaks.append(
                PredictedPeak(hkl=reflection.hkl_string, d=d, q=q)
            )
        return predicted_peaks
