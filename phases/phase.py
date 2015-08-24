# -*- coding: utf-8 -*-

from collections import namedtuple
import copy
import math

import scipy

import exceptions
from mapping.datadict import DataDict
from xrd.reflection import hkl_to_tuple
from phases.unitcell import UnitCell

class PhaseDataDict(DataDict):
    def __get__(self, obj, cls):
        new_dict = super().__get__(obj, cls)
        new_dict['unit_cell'] = obj.unit_cell.data_dict
        return new_dict

    def __set__(self, obj, new_dict):
        if 'unit_cell' in new_dict.keys():
            obj.unit_cell.data_dict = new_dict['unit_cell']
            del new_dict['unit_cell']
        return super().__set__(obj, new_dict)

class Phase():
    """A crystallographic phase that can be found in a Material."""
    name = None
    reflection_list = [] # Predicted peaks by crystallography
    spacegroup = ''
    scale_factor = 1
    unit_cell = UnitCell
    data_dict = PhaseDataDict(['scale_factor', 'u', 'v', 'w'])
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

    # @property
    # def data_dict(self):
    #     new_dict = {
    #         'scale_factor': self.scale_factor,
    #         'u': self.u,
    #         'v': self.v,
    #         'w': self.w,
    #         'unit_cell': self.unit_cell.data_dict
    #     }
    #     return new_dict

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

    def predicted_peak_positions(self, wavelength, unit_cell=None, scan=None):
        # Use current unit_cell if none is given
        if unit_cell is None:
            unit_cell = self.unit_cell
        PredictedPeak = namedtuple('PredictedPeak', ('hkl', 'd', 'two_theta'))
        predicted_peaks = []
        for reflection in self.reflection_list:
            # Only include reflection if it's within the scan's two-theta range
            if scan is not None:
                if not scan.contains_peak(reflection.two_theta_range):
                    continue
            # Calculate predicted position
            hkl = reflection.hkl
            d = unit_cell.d_spacing(hkl)
            radians = math.asin(wavelength/2/d)
            two_theta = 2*math.degrees(radians)
            predicted_peaks.append(
                PredictedPeak(reflection.hkl_string, d, two_theta)
            )
        return predicted_peaks
