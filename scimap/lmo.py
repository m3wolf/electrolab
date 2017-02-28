# -*- coding: utf-8 -*-
"""Sample definitions for lithium manganese oxide LiMn_2O_4"""

from matplotlib.colors import Normalize

from .xrd_map import XRDMap
from .phase import Phase
from .standards import Aluminum
from .unitcell import CubicUnitCell, TetragonalUnitCell
from .reflection import Reflection
from .fullprof_refinement import FullProfPhase, ProfileMatch


class CubicLMO(Phase):
    name = 'cubic LiMn2O4'
    unit_cell = CubicUnitCell(a=8.08)
    spacegroup = 'Fd-3m'
    fullprof_spacegroup = 'F D -3 M'
    diagnostic_hkl = '333'
    reflection_list = [
        Reflection('111', multiplicity=8, intensity=7.344,
                   qrange=(1.24, 1.38)),
        Reflection('220', multiplicity=12, intensity=0.036,
                   qrange=(2.11, 2.18)),
        Reflection('311', multiplicity=12, intensity=3.908,
                   qrange=(2.48, 2.59)),
        Reflection('222', multiplicity=8, intensity=1.024,
                   qrange=(2.61, 2.65)),
        Reflection('400', multiplicity=6, intensity=5.228,
                   qrange=(3.01, 3.12)),
        Reflection('331', multiplicity=24, intensity=1.301,
                   qrange=(3.32, 3.38)),
        Reflection('422', multiplicity=24, intensity=0.428,
                   qrange=(3.70, 3.76)),
        Reflection('511', multiplicity=8, intensity=2.111,
                   qrange=(3.89, 4.08)),
        Reflection('333', multiplicity=8, intensity=0.007,
                   qrange=(3.89, 4.08)),
        Reflection('440', multiplicity=12, intensity=3.162,
                   qrange=(4.29, 4.44)),
        Reflection('531', multiplicity=48, intensity=1.406,
                   qrange=(4.29, 4.44)),
        Reflection('442', multiplicity=24, intensity=0.070,
                   qrange=(4.50, 4.62)),
        # Reflection('620', multiplicity=24, intensity=0.031,
        #            qrange=(4.83, 4.91)),
        # Reflection('533', multiplicity=24, intensity=0.449,
        #            qrange=(4.91, 4.99)),
        # Reflection('622', multiplicity=24, intensity=0.185,
        #            qrange=(4.99, 5.07)),
    ]


class TetragonalLMO(Phase):
    unit_cell = TetragonalUnitCell()
    diagnostic_hkl = None
    reflection_list = [
        Reflection('000', qrange=(2.75, 2.82)),
    ]


class HighVPhase(FullProfPhase, CubicLMO):
    unit_cell = CubicUnitCell(a=8.053382)
    diagnostic_hkl = '333'
    reflection_list = [
        Reflection('333', qrange=(4.03, 4.08), multiplicity=8, intensity=0.081),
        Reflection('511', qrange=(4.03, 4.08), multiplicity=24, intensity=23.207),
        Reflection('440', qrange=(4.38, 4.44), multiplicity=12, intensity=39.180),
        Reflection('531', qrange=(4.59, 4.63), multiplicity=48, intensity=14.616),
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
        Reflection('333', qrange=(3.98, 4.03), multiplicity=8, intensity=0.081),
        Reflection('511', qrange=(3.98, 4.03), multiplicity=24, intensity=23.207),
        Reflection('440', qrange=(4.33, 4.38), multiplicity=12, intensity=39.180),
        Reflection('531', qrange=(4.54, 4.58), multiplicity=48, intensity=14.616),
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
