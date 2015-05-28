# -*- coding: utf-8 -*-

from matplotlib import colors
import numpy as np

class Material():
    """Describes a cathode material that may exist in multiple
    crystallographic phases.
    """
    def __init__(self, two_theta_range=(10, 80), scan_time=10, peak_list=[],
                 metric_peak=None, discharged_peak=None, charged_peak=None,
                 reliability_peak=None,
                 metric_normalizer=colors.Normalize(0, 1, clip=True),
                 reliability_normalizer=colors.Normalize(2.3, 4.5, clip=True),
                 *args, **kwargs):
        self.two_theta_range = two_theta_range
        self.scan_time = scan_time
        self.peak_list = peak_list
        self.metric_peak = metric_peak
        self.discharged_peak = discharged_peak
        self.charged_peak = charged_peak
        self.reliability_peak = reliability_peak
        self.metric_normalizer = metric_normalizer
        self.reliability_normalizer = reliability_normalizer

    def mapscan_metric(self, scan=None):
        """Contains the specifics of getting one number from each scan.
        To be overridden by actual Sample subclasses."""
        raise NotImplementedError

    def mapscan_reliability(self, scan):
        """
        Use peak area to determine how likely this scan is to represent
        sample rather than tape.
        """
        if self.reliability_peak:
            normalize = self.reliability_normalizer
            # Determine peak area for normalization
            df = scan.diffractogram()
            peakRange = self.peak_list[self.reliability_peak]
            peak = df.loc[peakRange[0]:peakRange[1], 'subtracted']
            peakArea = np.trapz(y=peak, x=peak.index)
            reliability = normalize(peakArea)
        else:
            reliability = 1
        return reliability

    def highlight_peaks(self, ax):
        """Highlight the peaks corresponding to each phase in this material."""
        for key, peak in self.peak_list.items():
            ax.axvspan(peak[0], peak[1], color='green', alpha=0.25)

class TwoPhaseMaterial(Material):
    def mapscan_metric(self, scan):
        """
        Compare the ratio of two peaks, one for discharged and one for
        charged material.
        """
        df = scan.diffractogram()
        # Get peak dataframes for integration
        peakDischarged = df.loc[
            self.peak_list[self.discharged_peak],
            'subtracted'
        ]
        peakCharged = df.loc[
            self.peak_list[self.charged_peak],
            'subtracted'
        ]
        # Integrate peaks
        areaCharged = np.trapz(y=peakCharged, x=peakCharged.index)
        areaDischarged = np.trapz(y=peakDischarged, x=peakDischarged.index)
        # Compare areas of the two peaks
        ratio = areaCharged/(areaCharged+areaDischarged)
        return ratio

class SolidSolutionMaterial(Material):
    def mapscan_metric(self, scan):
        """
        Return the 2θ difference of self.peak1 and self.peak2. Peak
        difference is used to overcome errors caused by shifter
        patterns.
        """
        df = scan.diffractogram()
        # Get the 2θ value of peak
        peak2 = self.peak_list[self.metric_peak]
        range2 = df.loc[peak2[0]:peak2[1], 'subtracted']
        theta2 = range2.argmax()
        return theta2

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

lmo_peak_list = {
    '111': (18, 20),
    '311': (36, 37),
    '222': (37.5, 39),
    '400': (44, 45),
    '400-charged': (44.5, 45.25),
    '400-discharged': (43.75, 44.5),
    'tetragonal': (39, 41),
    '331': (48, 49),
    '333': (55, 62),
    '511': (55, 62),
    '440': (62, 67),
    '531': (67, 70),
}

# Sample for mapping LiMn2O4 cathodes using peak position.
lmo_solid_solution = SolidSolutionMaterial(
    two_theta_range = (30, 50),
    scan_time = 200, # Seconds per frame
    peak_list = lmo_peak_list,
    metric_peak = '400',
    reliability_peak = '400',
    # Normalization ranges for the 2theta position of the target peak
    metric_normalizer = colors.Normalize(44, 45),
    reliability_normalizer = colors.Normalize(2.3, 4.5, clip=True),
)

# Material for mapping LiMn2O4 using peak area (two-phase mechanism)"""
lmo_two_phase = TwoPhaseMaterial(
    metric_normalizer = colors.Normalize(0, 1),
    reliability_normalizer = colors.Normalize(2.3, 4.5, clip=True),
    peak_list = lmo_peak_list,
    charged_peak = '400-charged',
    discharged_peak = '400-discharged',
    reliability_peak = '400',
)

# Sample for mapping LiMn2O4 in the low potential region from
# Li_1Mn_2O_4 to Mn_2O_4.
lmo_low_V = TwoPhaseMaterial(
    two_theta_range = (30, 50),
    peak_list = lmo_peak_list,
    discharged_peak = 'tetragonal',
    charged_peak = '400',
    reliability_peak = '400',
)

# MgMn_2O_4"""
mmo = TwoPhaseMaterial(
    scan_time = 2400,
    two_theta_range = (17.5, 37.5),
    peak_list = {
        'tetragonal1': (28, 30),
        'tetragonal2': (32, 34),
        'cubic': (35, 37),
    },
    discharged_peak = 'tetragonal1',
    charged_peak = 'cubic',
)
