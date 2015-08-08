# -*- coding: utf-8 -*-

from scipy.interpolate import UnivariateSpline

import plots
from xrd.peak import remove_peak_from_df

class NativeRefinement():

    def __init__(self, scan):
        self.scan = scan

    def refine_unit_cells(self):
        pass

    def refine_scale_factors(self):
        pass

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
