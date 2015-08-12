# -*- coding: utf-8 -*-

from phases.phase import Phase
from phases.unitcell import CubicUnitCell, TetragonalUnitCell
from xrd.reflection import Reflection

##################################################
# Sample definitions for lithium manganese oxide
# LiMn_2O_4
##################################################

class CubicLMO(Phase):
    name = 'cubic LiMn₂O₄'
    unit_cell = CubicUnitCell(a=8)
    spacegroup = 'Fd-3m'
    fullprof_spacegroup = 'F D -3 M'
    diagnostic_hkl = '311'
    reflection_list = [
        Reflection('111', multiplicity=8, intensity=7.344,
                   two_theta_range=(17.5, 19.5)),
        Reflection('220', multiplicity=12, intensity=0.036,
                   two_theta_range=(30, 31)),
        Reflection('311', multiplicity=12, intensity=3.908,
                   two_theta_range=(35.3, 37)),
        Reflection('222', multiplicity=8, intensity=1.024,
                   two_theta_range=(37.3, 38)),
        Reflection('400', multiplicity=6, intensity=5.228,
                   two_theta_range=(43.3, 45)),
        Reflection('331', multiplicity=24, intensity=1.301,
                   two_theta_range=(48, 49)),
        Reflection('422', multiplicity=24, intensity=0.428,
                   two_theta_range=(54, 55)),
        Reflection('511', multiplicity=8, intensity=2.111,
                   two_theta_range=(57, 60)),
        Reflection('333', multiplicity=8, intensity=0.007,
                   two_theta_range=(57, 60)),
        Reflection('440', multiplicity=12, intensity=3.162,
                   two_theta_range=(64, 66)),
        Reflection('531', multiplicity=48, intensity=1.406,
                   two_theta_range=(64, 66)),
        Reflection('442', multiplicity=24, intensity=0.070,
                   two_theta_range=(67, 70)),
        Reflection('620', multiplicity=24, intensity=0.031,
                   two_theta_range=(71, 74)),
        Reflection('533', multiplicity=24, intensity=0.449,
                   two_theta_range=(74, 75.5)),
        Reflection('622', multiplicity=24, intensity=0.185,
                   two_theta_range=(75.5, 77)),
    ]


class TetragonalLMO(Phase):
    unit_cell = TetragonalUnitCell()
    diagnostic_hkl = None
    reflection_list = [
        Reflection('000', (39.5, 40.5)),
    ]
