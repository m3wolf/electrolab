"""Module of different materials for XRD analysis."""

from matplotlib import colors
import numpy as np

from electrolab.mapping import BaseSample

class LMOSolidSolutionSample(BaseSample):
    """
    Sample for mapping LiMn2O4 cathodes using peak position.
    """
    two_theta_range = (30, 50)
    scan_time = 200 # Seconds per frame
    # Normalization ranges for the 2theta position of the target peak
    normalizer = colors.Normalize(44, 45)

    class XRDScan(BaseSample.XRDScan):
        peaks_by_hkl = {
            '111': (18, 20),
            '311': (36, 37),
            '222': (37.5, 39),
            '400': (44, 45),
            '331': (48, 49),
            '333': (55, 62),
            '511': (55, 62),
            '440': (62, 67),
            '531': (67, 70),
        }

        def metric(self):
            """
            Return the 2θ position of the (400) peak.
            """
            # Linear regression values determined by experiment
            df = self.diffractogram()
            # Get the 2θ value of peak 1
            peak1 = self.peaks_by_hkl['311']
            range1 = df.loc[peak1[0]:peak1[1], 'counts']
            theta1 = range1.argmax()
            # Get the 2θ value of peak 2
            peak2 = self.peaks_by_hkl['400']
            range2 = df.loc[peak2[0]:peak2[1], 'counts']
            theta2 = range2.argmax()
            # Return the result
            result = theta2-theta1
            return theta2

        def reliability(self):
            """
            Use peak area to determine how likely this scan is to represent
            sample rather than tape.
            """
            normalize = colors.Normalize(2.3, 4.5, clip=True)
            # Determine peak area for normalization
            df = self.diffractogram()
            peakRange = self.peaks_by_hkl['400']
            peak = df.loc[peakRange[0]:peakRange[1], 'subtracted']
            peakArea = np.trapz(y=peak, x=peak.index)
            # print(area)
            reliability = normalize(peakArea)
            return reliability


class LMOTwoPhaseSample(LMOSolidSolutionSample):
    """Sample for mapping LiMn2O4 using peak area (two-phase mechanism)"""
    normalizer = colors.Normalize(0, 1)
    class XRDScan(LMOSolidSolutionSample.XRDScan):
        def metric(self):
            """
            Compare the ratio of two peaks, one for discharged and one for
            charged material.
            """
            df = self.diffractogram()
            peakDischarged = df.loc[43.75:44.5, 'subtracted']
            peakCharged = df.loc[44.5:45.25, 'subtracted']
            areaCharged = np.trapz(y=peakCharged, x=peakCharged.index)
            areaDischarged = np.trapz(y=peakDischarged, x=peakDischarged.index)
            ratio = areaCharged/(areaCharged+areaDischarged)
            return ratio
