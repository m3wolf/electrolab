# -*- coding: utf-8 -*-

import pandas as pd
from pandas import Series
import numpy as np
import matplotlib.pyplot as plt

import exceptions
from xrd.tube import tubes, KALPHA2_RATIO
from peakfitting import Peak


class XRDPeak(Peak):
    """
    A peak in an X-ray diffractogram. May be composed of multiple
    overlapping subpeaks from different wavelengths.

    Arguments
    ---------
    - reflection : A diffraction reflection corresponding to this peak.
    - num_peaks (int) : How many subpeaks are present in this peak.
    - method (str) : Selects which peak shape to use. See
        peakfitting.Peak for valid choices
    """

    def __init__(self, reflection=None, num_peaks=2, method="pseudo-voigt",
                 tube=tubes["Cu"], *args, **kwargs):
        self.reflection = reflection
        self.tube = tube
        super().__init__(num_peaks=num_peaks, method=method, *args, **kwargs)

    def __repr__(self):
        name = "<{cls}: {angle}°>".format(
            cls=self.__class__.__name__,
            angle=self.center()
        )
        return name

    def guess_parameters(self, x, y):
        # Currently assumes two overlapping k-alpha peaks
        assert self.num_peaks == 2
        # Filter out data that is below half standard deviation of the whole set
        threshold = 0.5 * np.std(y)
        idx = np.where(y>threshold)
        peak_data = y[y > threshold]
        # Guess mean peak position based on weight average of x values
        if len(peak_data) == 0:
            msg = "No data exceeds background for peak."
            raise exceptions.PeakFitError(msg)
        mean_center = np.average(x, weights=y)

        # Convert average center to k-alpha1, k-alpha2
        center1, center2 = self.tube.split_angle_by_kalpha(mean_center)
        # Estimate kα₁, kα₂ heights
        height1 = y.max()
        height2 = height1 / 2
        # Guess initial parameters for the selected fitting method
        guess = [
            self.FitClass().initial_parameters(x=x,
                                               y=y,
                                               center=center1,
                                               height=height1),
            self.FitClass().initial_parameters(x=x,
                                               y=y,
                                               center=center2,
                                               height=height2),
        ]
        return guess

#     @property
#     def center_mean(self):
#         """Determine the average peak position based on fits."""
#         if len(self.fit_list) > 0:
#             total = sum([fit.center for fit in self.fit_list])
#             center = total/len(self.fit_list)
#         else:
#             center = None
#         return center

    @property
    def center_kalpha(self):
        """Determine the peak center based on relative intensities of kalpha1
        and kalpha2."""
        # Assumes only 2 peaks
        assert len(self.fit_list) == 2
        peak_alpha1 = self.fit_list[0].center
        peak_alpha2 = self.fit_list[1].center
        total = peak_alpha1 + (KALPHA2_RATIO * peak_alpha2)
        center = total / (1 + KALPHA2_RATIO)
        return center

#     def split_parameters(self, params):
#         """
#         Take a full list of parameters and divide it groups for each subpeak.
#         """
#         numFits = len(self.fit_list)
#         chunkSize = int(len(params)/numFits)
#         groups = []
#         for i in range(0, len(params), chunkSize):
#             groups.append(params[i:i+chunkSize])
#         return groups

#     def x_range(self):
#         """Return a range of x values over which this fit is reasonably defined."""
#         x = np.linspace(self.two_theta_range[0],
#                         self.two_theta_range[1],
#                         num=1000)
#         return x

#     def dataframe(self, background=None):
#         """Get a dataframe of the predicted peak fits."""
#         x = self.x_range()
#         y = np.zeros_like(x)
#         for fit in self.fit_list:
#             y += fit.evaluate(x)
#             if background is not None:
#                 # I don't know why I have to divide by 2 here but it works(!?)
#                 y += background(x)/2
#         return pandas.DataFrame(data=y, index=x, columns=['counts'])

#     def plot_overall_fit(self, ax=None, background=None):
#         df = self.dataframe(background=background)
#         if ax is None:
#             ax = plots.new_axes()
#         ax.plot(df.index, df.counts, label="overall fit")

#     def plot_fit(self, ax=None, background=None):
#         """Plot the subpeaks on the given axes. background(x) will be added to
#         each peak."""
#         x = self.x_range()
#         if ax is None:
#             ax = pyplot.gca()
#         for fit in self.fit_list:
#             y = fit.evaluate(x)
#             if background is not None:
#                 y = y+background(x)
#             ax.plot(x, y, label="fit subpeak")
