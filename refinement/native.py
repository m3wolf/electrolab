# -*- coding: utf-8 -*-

import numpy
from scipy.interpolate import UnivariateSpline
import pandas

import plots
from xrd.peak import remove_peak_from_df
from refinement.base import BaseRefinement

class NativeRefinement(BaseRefinement):
    def refine_unit_cells(self):
        pass

    def refine_scale_factors(self):
        # Scale factor is just the ratio of peak area of diagnostic reflection
        for phase in self.scan.phases:
            reflection = phase.diagnostic_reflection
            area = self.net_area(two_theta_range=reflection.two_theta_range)
            phase.scale_factor = area
        self.is_refined['scale_factors'] = True

    def refine_background(self):
        originalData = self.scan.diffractogram
        workingData = originalData.copy()
        # Remove pre-indexed peaks for background fitting
        phase_list = self.scan.phases + self.scan.background_phases
        for phase in phase_list:
            for reflection in phase.reflection_list:
                remove_peak_from_df(reflection, workingData)
        # Determine a background line from the noise without peaks
        self.spline = UnivariateSpline(
            x=workingData.index,
            y=workingData.counts,
            s=len(workingData.index)*25,
            k=4
        )
        # Extrapolate the background for the whole spectrum
        x = originalData.index
        originalData['background'] = self.spline(x)
        originalData['subtracted'] = originalData.counts - originalData.background
        return originalData

    def plot(self, ax=None):
        # Check if background has been fit
        spline = getattr(self, 'spline', None)
        if spline is not None:
            # Create new axes if necessary
            if ax is None:
                ax=plots.new_axes()
            # Plot background
            two_theta = self.scan.diffractogram.index
            background = self.spline(two_theta)
            ax.plot(two_theta, background)
        # Highlight peaks
        self.highlight_peaks(ax=ax)
        return ax

    def net_area(self, two_theta_range):
        """Area under the scan diffractogram after background subtraction."""
        fullDF = self.scan.diffractogram
        # Get peak dataframe for integration
        if self.scan.contains_peak(two_theta_range):
            peakDF = fullDF.loc[
                two_theta_range[0]:two_theta_range[1],
                'counts'
            ]
            # Subtract background
            background = pandas.Series(self.spline(peakDF.index),
                                       index=peakDF.index)
            netDF = peakDF - background
            # Integrate peak
            area = numpy.trapz(x=netDF.index, y=netDF)
        else:
            area = 0
        return area

    def highlight_peaks(self, ax):
        color_list = [
            'green',
            'blue',
            'red',
            'orange'
        ]
        # Highlight phases
        for idx, phase in enumerate(self.scan.phases):
            phase.highlight_peaks(ax=ax, color=color_list[idx])
        # Highlight background
        for phase in self.scan.background_phases:
            phase.highlight_peaks(ax=ax, color='grey')
        return ax
