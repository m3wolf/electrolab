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

    def __init__(self, scan):
        self.scan = scan
        if scan.filename is not None:
            self.basename = os.path.splitext(scan.filename)[0]

    def refine_background(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    def fwhm(self, phase=0):
        raise NotImplementedError
