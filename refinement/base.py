# -*- coding: utf-8 -*-

class BaseRefinement():
    # Status flags
    is_refined = {
        'background': False,
        'unit_cells': False,
        'scale_factors': False,
    }

    def __init__(self, scan):
        self.scan = scan

    def refine_background(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError
