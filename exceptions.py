# -*- coding: utf-8 -*-

class UnitCellError(ValueError):
    pass

class RefinementError(Exception):
    pass

class PeakFitError(RefinementError):
    pass
