# -*- coding: utf-8 -*-

from collections import namedtuple
import math

import scipy

import exceptions
from mapping.datadict import DataDict
from xrd.reflection import hkl_to_tuple

class Phase():
    """A crystallographic phase that can be found in a Material."""
    name = None
    reflection_list = [] # Predicted peaks by crystallography
    spacegroup = ''
    scale_factor = 1
    data_dict = DataDict(['scale_factor'])

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
