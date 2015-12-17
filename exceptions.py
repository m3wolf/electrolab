# -*- coding: utf-8 -*-

class ReadCurrentError(Exception):
    """Cannot read a current from electrochemistry cycling file."""
    pass

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

class DivergenceError(RefinementError):
    pass

class PCRFileError(RefinementError):
    pass

class NoReflectionsError(RefinementError):
    """The refinement has I(obs) = 0. (Do you really have reflections?)"""
    def __str__(self):
        msg = "I(obs) = 0. Do you really have reflections? {}".format(self.args)
        return msg

class FileFormatError(ValueError):
    pass

## X-ray microscopy error
class FrameFileNotFound(IOError):
    pass

class GroupKeyError(KeyError):
    pass

class FileExistsError(IOError):
    pass

class DatasetExistsError(RuntimeError):
    pass

class NoParticleError(Exception):
    pass
