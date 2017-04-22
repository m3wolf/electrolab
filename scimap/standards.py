# -*- coding: utf-8 -*-

from .phase import Phase
from .unitcell import HexagonalUnitCell, CubicUnitCell
from .reflection import Reflection


class Corundum(Phase):
    name = 'corundum'
    unit_cell = HexagonalUnitCell(a=4.75, c=12.982)
    spacegroup = 'R-3c'
    fullprof_spacegroup = 'R -3 C'
    diagnostic_hkl = '104'
    reflection_list = [
        Reflection('012', qrange=(1.76, 1.90)),
        Reflection('104', qrange=(2.38, 2.52)),
        Reflection('110', qrange=(2.59, 2.72)),
        Reflection('006', qrange=(2.85, 2.95)),
        Reflection('113', qrange=(2.95, 3.05)),
        Reflection('024', qrange=(3.57, 3.70)),
        Reflection('116', qrange=(3.83, 4.01)),
        Reflection('211', qrange=(4.02, 4.09)),
        Reflection('214', qrange=(4.11, 4.25)),
        Reflection('300', qrange=(4.45, 4.50)),
        Reflection('125', qrange=(4.55, 4.60)),
        Reflection('208', qrange=(4.85, 4.97)),
        Reflection('1 0 10', qrange=(5, 5.15)),
    ]


class Aluminum(Phase):
    name = 'aluminum'
    unit_cell = CubicUnitCell(a=4.05)
    diagnostic_hkl = '111'
    reflection_list = [
        Reflection('111', qrange=(2.61, 2.72)),
        Reflection('200', qrange=(3.02, 3.12)),
        Reflection('220', qrange=(4.29, 4.41)),
        Reflection('311', qrange=(5.07, 5.24)),
        Reflection('222', qrange=(5.29, 5.45)),
    ]
