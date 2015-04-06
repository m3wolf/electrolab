"""Module of different materials for XRD analysis."""

from matplotlib import colors
import numpy as np

from electrolab.mapping import BaseSample

class LMOSample(BaseSample):
    """
    Sample for mapping LiMn2O4 cathodes.
    """
    two_theta_range = (30, 50)
    scan_time = 600 # 10 minutes per frame
    peaks_by_hkl = {
        '111': (18, 20),
        '311': (35, 38),
        '222': (37.5, 39),
        '400': (42, 47),
        '331': (47, 50),
        '333': (55, 62),
        '511': (55, 62),
        '440': (62, 67),
        '531': (67, 70),
    }
    # Normalization ranges for the 2theta position of the target peak
    normalizer = colors.Normalize(44, 45)

    class XRDScan(BaseSample.XRDScan):

        def metric(self):
            """
            Compare the 2θ positions of two peaks. Using two peaks may correct
            for differences is sample height on the instrument.
            """
            # Linear regression values determined by experiment
            df = self.diffractogram()
            # Get the 2θ value of peak 1
            peak1 = LMOSample.peaks_by_hkl['311']
            range1 = df.loc[peak1[0]:peak1[1], 'counts']
            theta1 = range1.argmax()
            # Get the 2θ value of peak 2
            peak2 = LMOSample.peaks_by_hkl['400']
            range2 = df.loc[peak2[0]:peak2[1], 'counts']
            theta2 = range2.argmax()
            # Return the result
            result = theta2-44
            return theta2

        def reliability(self):
            """
            Measure background fluorescence to detect tape.
            """
            # Adjust the normalization range based on the scanning parameters
            normalizeMin = 75 # Starting value
            normalizeMin = normalizeMin * self.sample.scan_time/400
            normalizeMin = normalizeMin *self.sample.collimator**2 / 0.5**2
            normalizeMax = 150 # Starting value
            normalizeMax = normalizeMax * self.sample.scan_time/400
            normalizeMax = normalizeMax *self.sample.collimator**2 / 0.5**2
            normalize = colors.Normalize(normalizeMin, normalizeMax, clip=True)
            # Calculate reliability from background
            diffractogram = self.diffractogram()
            background = diffractogram.loc[41:44, 'counts'].mean()
            reliability = normalize(background)
            return reliability


class LMOAreaSample(LMOSample):
    """Similar to LMOSample but using the ratios of peak areas as the metric"""
    normalizer = colors.Normalize(0, 1)
    class XRDScan(LMOSample.XRDScan):
        def metric(self):
            """
            Compare the ratio of two peaks, one for discharged and one for
            charged material.
            """
            df = self.diffractogram()
            peakDischarged = df.loc[43.75: 44.5, 'subtracted']
            peakCharged = df.loc[44.5:45.25, 'subtracted']
            areaCharged = np.trapz(y=peakCharged, x=peakCharged.index)
            areaDischarged = np.trapz(y=peakDischarged, x=peakDischarged.index)
            ratio = areaCharged/(areaCharged+areaDischarged)
            return ratio
