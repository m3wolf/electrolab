# -*- coding: utf-8 -*-
"""Sample definitions for lithium manganese oxide LiMn_2O_4"""

from matplotlib.colors import Normalize

from .map import XRDMap
from .phase import Phase
from .standards import Aluminum
from .unitcell import CubicUnitCell, TetragonalUnitCell
from .reflection import Reflection
from refinement.fullprof import FullProfPhase, ProfileMatch


class CubicLMO(Phase):
    name = 'cubic LiMn2O4'
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
                   two_theta_range=(63.5, 66)),
        Reflection('531', multiplicity=48, intensity=1.406,
                   two_theta_range=(63.5, 66)),
        Reflection('442', multiplicity=24, intensity=0.070,
                   two_theta_range=(67, 69)),
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


class HighVPhase(FullProfPhase, CubicLMO):
    unit_cell = CubicUnitCell(a=8.053382)
    diagnostic_hkl = '333'
    reflection_list = [
        Reflection('333', (59.3, 60.1), multiplicity=8, intensity=0.081),
        Reflection('511', (59.3, 60.1), multiplicity=24, intensity=23.207),
        Reflection('440', (65.1, 66.0), multiplicity=12, intensity=39.180),
        Reflection('531', (68.5, 69.3), multiplicity=48, intensity=14.616),
    ]
    u = -0.139195
    v = 0.198405
    w = 0.008828
    I_g = -0.033578
    eta = 0.386
    x = 0.013
    isotropic_temp = -2.1903
    scale_factor = 0.05


class MidVPhase(FullProfPhase, CubicLMO):
    unit_cell = CubicUnitCell(a=8.12888)
    diagnostic_hkl = '333'
    reflection_list = [
        Reflection('333', (58.5, 59.3), multiplicity=8, intensity=0.081),
        Reflection('511', (58.5, 59.3), multiplicity=24, intensity=23.207),
        Reflection('440', (64.2, 65.1), multiplicity=12, intensity=39.180),
        Reflection('531', (67.7, 68.5), multiplicity=48, intensity=14.616),
    ]
    scale_factor = 0.05
    u = -0.139195
    v = 0.198405
    w = 0.033169
    I_g = 0
    eta = 0.209160
    x = 0.013
    isotropic_temp = -3.7293


# Prepare materials with new reflections
class MidV440Phase(MidVPhase):
    diagnostic_hkl = '440'


class HighV440Phase(HighVPhase):
    diagnostic_hkl = '440'


class MidV531Phase(MidVPhase):
    diagnostic_hkl = '531'


class HighV531Phase(HighVPhase):
    diagnostic_hkl = '531'


class LMORefinement(ProfileMatch):
    bg_coeffs = [0.409, 14.808, -14.732, -10.292, 34.249, -28.046]
    zero = -0.001360
    displacement = 0.000330
    transparency = -0.008100


# Define a new class for mapping the two-phase plateau
class LMOPlateauMap(XRDMap):
    scan_time = 300
    two_theta_range = (55, 70)
    phases = [HighVPhase, MidVPhase]
    background_phases = [Aluminum]
    phase_ratio_normalizer = Normalize(0, 0.7, clip=True)
    reliability_normalizer = Normalize(0.7, 2, clip=True)
