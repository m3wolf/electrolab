# -*- coding: utf-8 -*-

from phases.phase import Phase
from phases.unitcell import HexagonalUnitCell
from xrd.reflection import Reflection

# Corundum standard
class Corundum(Phase):
    unit_cell = HexagonalUnitCell(a=4.75, c=12.982)
    name='corundum'
    diagnostic_hkl = '104'
    reflection_list = [
        Reflection((25, 27), '012'),
        Reflection((34, 36), '104'),
        Reflection((37, 39), '110'),
        Reflection((41, 42.5), '006'),
        Reflection((42.5, 44), '113'),
        Reflection((52, 54), '024'),
        Reflection((56, 59), '116'),
    ]

class Aluminum(Phase):
    name = 'aluminum'
    diagnostic_hkl = '111'
    reflection_list = [
        Reflection((37.3, 39), '111'),
        Reflection((43.5, 45), '200'),
        Reflection((63.5, 65.5), '220'),
        Reflection((77, 80), '311'),
        Reflection((81, 84), '222'),
    ]
