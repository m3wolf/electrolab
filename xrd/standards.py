# -*- coding: utf-8 -*-

from .phase import Phase
from .unitcell import HexagonalUnitCell, CubicUnitCell
from .reflection import Reflection

# Corundum standard
class Corundum(Phase):
    name = 'corundum'
    unit_cell = HexagonalUnitCell(a=4.75, c=12.982)
    spacegroup = 'R-3c'
    fullprof_spacegroup = 'R -3 C'
    diagnostic_hkl = '104'
    reflection_list = [
        Reflection('012', (25, 27)),
        Reflection('104', (34, 36)),
        Reflection('110', (37, 39)),
        Reflection('006', (41, 42.5)),
        Reflection('113', (42.5, 44)),
        Reflection('024', (52, 54)),
        Reflection('116', (56, 59)),
    ]


class Aluminum(Phase):
    name = 'aluminum'
    unit_cell = CubicUnitCell(a=4.05)
    diagnostic_hkl = '111'
    reflection_list = [
        Reflection('111', (37.3, 39)),
        Reflection('200', (43.5, 45)),
        Reflection('220', (63.5, 65.5)),
        Reflection('311', (77, 80)),
        Reflection('222', (81, 84)),
    ]
