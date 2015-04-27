"""Module of different materials for XRD analysis."""

from matplotlib import colors
import numpy as np

from electrolab.mapping import SolidSolutionSample, TwoPhaseSample


##################################################
# Sample definitions for lithium manganese oxide
# LiMn_2O_4
##################################################

lmo_peak_list = {
    '111': (18, 20),
    '311': (36, 37),
    '222': (37.5, 39),
    '400': (44, 45),
    '400-charged': (44.5, 45.25),
    '400-discharged': (43.75, 44.5),
    '331': (48, 49),
    '333': (55, 62),
    '511': (55, 62),
    '440': (62, 67),
    '531': (67, 70),
}

class LMOSolidSolution(SolidSolutionSample):
    """
    Sample for mapping LiMn2O4 cathodes using peak position.
    """
    two_theta_range = (30, 50)
    scan_time = 200 # Seconds per frame
    peak_list = lmo_peak_list
    metric_peak = '400'
    reliability_peak = '400'
    # Normalization ranges for the 2theta position of the target peak
    metric_normalizer = colors.Normalize(44, 45)
    reliability_normalizer = colors.Normalize(2.3, 4.5, clip=True)


class LMOTwoPhase(TwoPhaseSample):
    """Sample for mapping LiMn2O4 using peak area (two-phase mechanism)"""
    metric_normalizer = colors.Normalize(0, 1)
    reliability_normalizer = colors.Normalize(2.3, 4.5, clip=True)
    peak_list = lmo_peak_list
    charged_peak = '400-charged'
    discharged_peak = '400-discharged'
    reliability_peak = '400'
