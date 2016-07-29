# -*- coding: utf-8 -*-
"""Sample definitions for nickel-cobalt-aluminum oxide
LiNi_{0.8}Co_{0.15}Al_{0.05}O_2

"""

from matplotlib.colors import Normalize

from .map import XRDMap
from .phase import Phase
from .standards import Aluminum
from .unitcell import CubicUnitCell, TetragonalUnitCell, HexagonalUnitCell
from .reflection import Reflection
from refinement.fullprof import FullProfPhase, ProfileMatch


class NCA(Phase):
    name = 'NCA'
    unit_cell = HexagonalUnitCell(a=2.86687, c=14.18385)
    spacegroup = 'R-3m'
    fullprof_spacegroup = 'R -3 M'
    diagnostic_hkl = '113'
    reflection_list = [
        Reflection('003', qrange=(1.30, 1.50)),
        Reflection('101', qrange=(2.50, 2.65)),
        Reflection('006', qrange=(2.60, 2.80)),
        Reflection('102', qrange=(2.62, 2.80)),
        Reflection('104', qrange=(3.05, 3.25)),
        Reflection('105', qrange=(3.30, 3.60)),
        Reflection('009', qrange=(3.90, 4.20)),
        Reflection('107', qrange=(3.90, 4.20)),
        Reflection('108', qrange=(4.30, 4.60)),
        Reflection('110', qrange=(4.35, 4.60)),
        Reflection('113', qrange=(4.50, 4.75)),
    ]

# Define a new class for mapping the transition
class NCAMap(XRDMap):
    scan_time = 300
    two_theta_range = (53, 73)
    phases = [NCA]
    background_phases = [Aluminum]
    # phase_ratio_normalizer = Normalize(0, 0.7, clip=True)
    # reliability_normalizer = Normalize(0.7, 2, clip=True)
