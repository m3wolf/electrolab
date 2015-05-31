# -*- coding: utf-8 -*-

from matplotlib import colors
import numpy as np

from electrolab.xrd import Phase, Reflection

class Material():
    """Describes a cathode material that may exist in multiple
    crystallographic phases.
    """
    two_theta_range = (10, 80)
    scan_time = 10
    phase_list = []
    background_phases = []
    metric_normalizer = colors.Normalize(0, 1, clip=True)
    reliability_normalizer = colors.Normalize(0, 1, clip=True)
    # def __init__(self, two_theta_range=(10, 80), scan_time=10,
    #              metric_peak=None, discharged_peak=None, charged_peak=None,
    #              reliability_peak=None,
    #              phase_list=[], background_phases=[],
    #              metric_normalizer=colors.Normalize(0, 1, clip=True),
    #              reliability_normalizer=colors.Normalize(2.3, 4.5, clip=True),
    #              *args, **kwargs):
        # self.two_theta_range = two_theta_range
        # self.scan_time = scan_time
        # self.metric_peak = metric_peak
        # self.discharged_peak = discharged_peak
        # self.charged_peak = charged_peak
        # self.reliability_peak = reliability_peak
        # self.phase_list = phase_list
        # self.background_phases = background_phases
        # self.metric_normalizer = metric_normalizer
        # self.reliability_normalizer = reliability_normalizer

    def mapscan_metric(self, scan=None):
        """Contains the specifics of getting one number from each scan.
        To be overridden by actual Sample subclasses."""
        raise NotImplementedError

    def mapscan_reliability(self, scan):
        """
        See how much signal we have compared to background.
        """
        signal = 0
        # Calculate signals
        for phase in self.phase_list:
            two_theta_range = phase.diagnostic_reflection.two_theta_range
            signal += scan.peak_area(two_theta_range)
        # Calculate background signal
        background = 0
        for phase in self.background_phases:
            two_theta_range = phase.diagnostic_reflection.two_theta_range
            background += scan.peak_area(two_theta_range)
        reliability = signal / (signal+background)
        return reliability

    def highlight_peaks(self, ax):
        """Highlight the peaks corresponding to each phase in this material."""
        color_list = [
            'green',
            'blue',
            'red',
            'orange'
        ]
        alpha = 0.15
        for i, phase in enumerate(self.phase_list):
            for reflection in phase.reflection_list:
                two_theta = reflection.two_theta_range
                ax.axvspan(two_theta[0], two_theta[1], color=color_list[i], alpha=alpha)
        # Highlight background phases (sample stage, etc)
        for phase in self.background_phases:
            for reflection in phase.reflection_list:
                two_theta = reflection.two_theta_range
                ax.axvspan(two_theta[0], two_theta[1], color='0.8', alpha=1)

class TwoPhaseMaterial(Material):

    def mapscan_metric(self, scan):
        """
        Compare the ratio of two peaks, one for discharged and one for
        charged material.
        """
        # Integrate peaks
        chargedPeak = self.phase_list[0].diagnostic_reflection.two_theta_range
        areaCharged = scan.peak_area(chargedPeak)
        dischargedPeak = self.phase_list[1].diagnostic_reflection.two_theta_range
        areaDischarged = scan.peak_area(dischargedPeak)
        # Compare areas of the two peaks
        ratio = areaCharged/(areaCharged+areaDischarged)
        return ratio

class IORMaterial(TwoPhaseMaterial):
    """One-off material for submitting an image of the Image of Research
    competition at UIC."""
    metric_normalizer = colors.Normalize(0, 1, clip=True)
    reliability_normalizer = colors.Normalize(2.3, 4.5, clip=True)
    charged_peak = '331'
    discharged_peak = '400'
    reliability_peak = '400'
    def mapscan_metric(self, scan):
        area = self.peak_area(scan, self.peak_list[self.charged_peak])
        return area

class SolidSolutionMaterial(Material):
    def mapscan_metric(self, scan):
        """
        Return the 2θ difference of self.peak1 and self.peak2. Peak
        difference is used to overcome errors caused by shifter
        patterns.
        """
        main_phase = self.phase_list[0]
        two_theta_range = main_phase.diagnostic_reflection.two_theta_range
        metric = scan.peak_position(two_theta_range)
        return metric

class DummyMaterial(Material):
    def mapscan_metric(self, scan=None):
        # Just return the distance from bottom left to top right
        p = self.cube_coords[0]
        rows = self.sample.rows
        r = p/2/rows + 0.5
        return r


## Standard material definitions below

##################################################
# Sample definitions for lithium manganese oxide
# LiMn_2O_4
##################################################

#     '400-charged': (44.5, 45.25),
#     '400-discharged': (43.75, 44.5),

# lmo_peak_list = {
#     'null': (0, 0),
#     '111': (18, 20),
#     '311': (36, 37),
#     '222': (37.5, 39),
#     '400': (44, 45),
#     '400-charged': (44.5, 45.25),
#     '400-discharged': (43.75, 44.5),
#     'tetragonal': (39, 41),
#     '331': (49, 49.8),
#     '333': (55, 62),
#     '511': (55, 62),
#     '440': (62, 67),
#     '531': (67, 70),
# }

lmo_cubic_phase = Phase(
    crystal_system = "cubic",
    diagnostic_reflection = '311',
    reflection_list = [
        Reflection((18, 20), '111'),
        Reflection((35.3, 36.5), '311'),
        Reflection((37.3, 38), '222'),
        Reflection((43.3, 45), '400'),
        Reflection((48, 49), '331'),
        Reflection((55, 62), '333'),
        Reflection((55, 62), '511'),
        Reflection((62, 67), '440'),
        Reflection((67, 70), '531'),
    ]
)

lmo_tetragonal_phase = Phase(
    crystal_system = "cubic",
    diagnostic_reflection = '000',
    reflection_list = [
        Reflection((39.5, 40.5), '000'),
    ]
)

# Currently not indexed properly
aluminum_phase = Phase(
    crystal_system = None,
    diagnostic_reflection = '000',
    reflection_list = [
        Reflection((37.3, 38), '000'),
        Reflection((43.5, 45), '001')
    ]
)

# Material for mapping LiMn2O4 cathodes using peak position.
# lmo_solid_solution_material = SolidSolutionMaterial(
class LMOSolidSolutionMaterial(SolidSolutionMaterial):
    two_theta_range = (30, 50)
    scan_time = 200 # Seconds per frame
    phase_list = [lmo_cubic_phase]
    background_phases = [aluminum_phase]
    metric_peak = '400'
    reliability_peak = '400'
    # Normalization ranges for the 2theta position of the target peak
    metric_normalizer = colors.Normalize(44, 45, clip=True)
    reliability_normalizer = colors.Normalize(2.3, 4.5, clip=True)


# Material for mapping LiMn2O4 using peak area (two-phase mechanism)"""
#lmo_two_phase_material = TwoPhaseMaterial(
class LMOTwoPhaseMaterial(TwoPhaseMaterial):
    metric_normalizer = colors.Normalize(0.8, 1.2, clip=True)
    reliability_normalizer = colors.Normalize(0, 1, clip=True)
    phase_list = [lmo_cubic_phase, lmo_tetragonal_phase]
    background_phases = [aluminum_phase]
    charged_peak = '400-charged'
    discharged_peak = '400-discharged'
    reliability_peak = '400'


# ior_material = IORMaterial(
#     metric_normalizer = colors.Normalize(0, 1, clip=True),
#     reliability_normalizer = colors.Normalize(2.3, 4.5, clip=True),
#     charged_peak = '331',
#     discharged_peak = '400',
#     reliability_peak = '400',
# )

# Sample for mapping LiMn2O4 in the low potential region from
# Li_1Mn_2O_4 to Mn_2O_4.
# lmo_lowV_material = TwoPhaseMaterial(
class LMOLowVMaterial(TwoPhaseMaterial):
    two_theta_range = (30, 50),
    phase_list = [lmo_cubic_phase, lmo_tetragonal_phase],
    background_phases = [aluminum_phase],
    discharged_peak = 'tetragonal',
    charged_peak = '400',
    reliability_peak = '400',


# MgMn_2O_4
mmo_tetragonal_phase = Phase(
    crystal_system = "tetragonal",
    reflection_list = [
        Reflection((28, 30), '000'),
        Reflection((32, 34), '000'),
    ]
)

mmo_cubic_phase = Phase(
    crystal_system = "cubic",
    reflection_list = [
        Reflection((35, 37), '000'),
    ]
)

# mmo_material = TwoPhaseMaterial(
class MMOMaterial(TwoPhaseMaterial):
    scan_time = 2400
    two_theta_range = (17.5, 37.5)
    phase_list = [mmo_cubic_phase, mmo_tetragonal_phase]
    # peak_list = {
    #     'tetragonal1': (28, 30),
    #     'tetragonal2': (32, 34),
    #     'cubic': (35, 37),
    # },
    discharged_peak = 'tetragonal1'
    charged_peak = 'cubic'
