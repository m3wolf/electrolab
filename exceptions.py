# -*- coding: utf-8 -*-

class UnitCellError(ValueError):
    pass

class RefinementError(Exception):
    pass

class PeakFitError(RefinementError):
    pass

class EmptyRefinementError(RefinementError):
    pass

class SingularMatrixError(RefinementError):
    def __init__(self, param):
        self.param = param

    def __str__(self):
        msg = "Singular matrix while refining {param}".format(param=self.param)
        return msg

class NoReflectionsError(RefinementError):
    """The refinement has I(obs) = 0. (Do you really have reflections?)"""
    def __str__(self):
        msg = "I(obs) = 0. Do you really have reflections? {}".format(self.args)
        return msg

class FileFormatError(ValueError):
    pass
