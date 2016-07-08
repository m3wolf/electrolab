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

    def plot(self):
        raise NotImplementedError

    def fwhm(self, phase=0):
        raise NotImplementedError
