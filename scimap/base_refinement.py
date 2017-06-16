# -*- coding: utf-8 -*-

import os


class BaseRefinement():
    # Status flags
    is_refined = {
        'displacement': False,
        'background': False,
        'peak_widths': False,
        'unit_cells': False,
        'scale_factors': False,
    }

    def __init__(self, phases=[], background_phases=[]):
        self.phases = phases
        self.background_phases = background_phases

    def refine_background(self):
        raise NotImplementedError

    def refine_unit_cells(self):
        raise NotImplementedError

    def scale_factors(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    def refine_peak_widths(self, scattering_lengths, intensities):
        raise NotImplementedError

    def phase_weights(self):
        raise NotImplementedError
