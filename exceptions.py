# -*- coding: utf-8 -*-

class UnitCellError(ValueError):
    pass

class RefinementError(Exception):
    pass

class PeakFitError(RefinementError):
    pass

class FileFormatError(ValueError):
    pass

class EmptyRefinementError(RefinementError):
    pass
