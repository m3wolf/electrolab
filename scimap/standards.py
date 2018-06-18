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
        # HKL indices taken from ICDD entry 46-1212
        Reflection('012', intensity=45,qrange=(1.76, 1.90)),
        Reflection('104', intensity=100, qrange=(2.38, 2.52)),
        Reflection('110', intensity=21, qrange=(2.59, 2.72)),
        Reflection('006', intensity=2, qrange=(2.85, 2.95)),
        Reflection('113', intensity=66, qrange=(2.95, 3.05)),
        Reflection('202', intensity=1),
        Reflection('024', intensity=34, qrange=(3.57, 3.70)),
        Reflection('116', intensity=89, qrange=(3.83, 4.01)),
        Reflection('211', intensity=1, qrange=(4.02, 4.09)),
        Reflection('122', intensity=2),
        Reflection('018', intensity=14),
        Reflection('214', intensity=23, qrange=(4.11, 4.25)),
        Reflection('300', intensity=27, qrange=(4.45, 4.50)),
        Reflection('125', intensity=1, qrange=(4.55, 4.60)),
        Reflection('208', intensity=2, qrange=(4.85, 4.97)),
        Reflection('1 0 10', intensity=29, qrange=(5, 5.15)),
        Reflection('119', intensity=12),
        Reflection('217', intensity=1),
        Reflection('220', intensity=2),
        Reflection('306', intensity=1),
        Reflection('223', intensity=3),
        Reflection('131', intensity=0),
        Reflection('312', intensity=2),
        Reflection('128', intensity=3),
        Reflection('0 2 10', intensity=9),
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
