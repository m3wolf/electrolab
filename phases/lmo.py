# -*- coding: utf-8 -*-

from phases.phase import Phase
from phases.unitcell import CubicUnitCell, TetragonalUnitCell
from xrd.reflection import Reflection

##################################################
# Sample definitions for lithium manganese oxide
# LiMn_2O_4
##################################################

class CubicLMO(Phase):
    unit_cell = CubicUnitCell(a=8)
    diagnostic_hkl = '311'
    reflection_list = [
        Reflection((17.5, 19.5), '111'),
        Reflection((35.3, 37), '311'),
        Reflection((37.3, 38), '222'),
        Reflection((43.3, 45), '400'),
        Reflection((48, 49), '331'),
        Reflection((57, 60), '333'),
        Reflection((57, 60), '511'),
        Reflection((64, 66), '440'),
        Reflection((67, 69), '531'),
    ]


class TetragonalLMO(Phase):
    unit_cell = TetragonalUnitCell()
    diagnostic_hkl = None
    reflection_list = [
        Reflection((39.5, 40.5), '000'),
    ]
