# -*- coding: utf-8 -*-

from matplotlib import colors
import numpy as np

from electrolab import xrd
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
            if scan.contains_peak(peak=two_theta_range):
                signal += scan.peak_area(two_theta_range)
        # Calculate background signal
        background = 0
        for phase in self.background_phases:
            two_theta_range = phase.diagnostic_reflection.two_theta_range
            background += scan.peak_area(two_theta_range)
        totalIntensity = signal + background
        intensityModifier = colors.Normalize(0.15, 0.5, clip=True)(totalIntensity)
        reliability = intensityModifier * signal / (signal+background)
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

lmo_cubic_phase = Phase(
    unit_cell = xrd.CubicUnitCell(a=8),
    diagnostic_reflection = '311',
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
)

lmo_tetragonal_phase = Phase(
    # crystal_system = "cubic",
    diagnostic_reflection = None,
    reflection_list = [
        Reflection((39.5, 40.5), '000'),
    ]
)

# Currently not indexed properly
aluminum_phase = Phase(
    name = 'aluminum',
    # crystal_system = None,
    diagnostic_reflection = '111',
    reflection_list = [
        Reflection((37.3, 39), '111'),
        Reflection((43.5, 45), '200'),
        Reflection((63.5, 65.5), '220'),
        Reflection((77, 80), '311'),
        Reflection((81, 84), '222'),
    ]
)

# 304 Stainless steel. PDF2 card: 00-033-0397
stainless_steel_phase = Phase(
    # crystal_system = 'cubic',
    reflection_list = [
        Reflection((42, 46), '111'),
        Reflection((49, 53), '200'),
        Reflection((72, 78), '220'),
    ]
)
class StainlessSteelMaterial(Material):
    phase_list = [stainless_steel_phase]

# Corundum standard
class CorundumMaterial(Material):
    phase_list = [Phase(
        # crystal_system = 'rhombohedral',
        unit_cell = xrd.HexagonalUnitCell(a=4.75, c=12.982),
        reflection_list = [
            Reflection((25, 27), '012'),
            Reflection((34, 36), '104'),
            Reflection((37, 39), '110'),
            Reflection((41, 42.5), '006'),
            Reflection((42.5, 44), '113'),
            Reflection((52, 54), '024'),
            Reflection((56, 59), '116'),
        ]
    )]


# Material for mapping LiMn2O4 cathodes using peak position.
class LMOSolidSolutionMaterial(SolidSolutionMaterial):
    two_theta_range = (30, 50)
    scan_time = 200 # Seconds per frame
    phase_list = [lmo_cubic_phase]
    background_phases = [aluminum_phase]
    # Normalization ranges for the 2theta position of the target peak
    metric_normalizer = colors.Normalize(44, 45, clip=True)
    reliability_normalizer = colors.Normalize(0.4, 0.8, clip=True)


# Material for mapping LiMn2O4 using peak area (two-phase mechanism)"""
class LMOTwoPhaseMaterial(TwoPhaseMaterial):
    metric_normalizer = colors.Normalize(0.8, 1.2, clip=True)
    reliability_normalizer = colors.Normalize(0, 1, clip=True)
    phase_list = [lmo_cubic_phase, lmo_tetragonal_phase]
    background_phases = [aluminum_phase]


# Sample for mapping LiMn2O4 in the low potential region from
# Li_1Mn_2O_4 to Mn_2O_4.
class LMOLowVMaterial(TwoPhaseMaterial):
    two_theta_range = (30, 50),
    phase_list = [lmo_cubic_phase, lmo_tetragonal_phase],
    background_phases = [aluminum_phase],
    discharged_peak = 'tetragonal',
    charged_peak = '400',
    reliability_peak = '400',


##################################################
# Sample definitions for lithium manganese oxide
# MgMn_2O_4
##################################################

mmo_tetragonal_phase = Phase(
    # crystal_system = "tetragonal",
    reflection_list = [
        Reflection((28, 30), '000'),
        Reflection((32, 34), '000'),
    ]
)

mmo_cubic_phase = Phase(
    # crystal_system = "cubic",
    diagnostic_reflection = None,
    reflection_list = [
        Reflection((35, 37), '000'),
    ]
)


class MMOMaterial(TwoPhaseMaterial):
    scan_time = 2400
    two_theta_range = (17.5, 37.5)
    phase_list = [mmo_cubic_phase, mmo_tetragonal_phase]
    background_phases = [stainless_steel_phase]
    discharged_peak = 'tetragonal1'
    charged_peak = 'cubic'
