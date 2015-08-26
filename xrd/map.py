# -*- coding: utf-8 --*

from matplotlib.colors import Normalize

import exceptions
from mapping.map import Map, display_progress
from xrd.mapscan import XRDMapScan

class XRDMap(Map):
    cell_class = XRDMapScan
    cell_parameter_normalizer = None
    phase_ratio_normalizer = None
    def set_metric_phase_ratio(self, phase_idx=0):
        """Set the plotting metric as the proportion of given phase."""
        for scan in display_progress(self.scans, 'Calculating metrics'):
            phase_scale = scan.phases[phase_idx].scale_factor
            total_scale = sum([phase.scale_factor for phase in scan.phases])
            scan.metric = phase_scale/total_scale

    def plot_phase_ratio(self, phase_idx=0, *args, **kwargs):
        """Plot a map of the ratio of the given phase index to all the phases"""
        self.set_metric_phase_ratio(phase_idx=0)
        # Determine normalization range
        if self.phase_ratio_normalizer is None:
            metrics = [scan.metric for scan in self.scans]
            self.metric_normalizer = Normalize(min(metrics),
                                               max(metrics),
                                               clip=True)
        else:
            self.metric_normalizer = self.phase_ratio_normalizer
        # Plot the map
        return self.plot_map(*args, **kwargs)

    def set_metric_cell_parameter(self, parameter='a', phase_idx=0):
        for scan in display_progress(self.scans, 'Calculating cell parameters'):
            phase = scan.phases[phase_idx]
            scan.metric = getattr(phase.unit_cell, parameter)

    def plot_cell_parameter(self, parameter='a', phase_idx=0, *args, **kwargs):
        self.set_metric_cell_parameter(parameter, phase_idx)
        # Determine normalization range
        if self.cell_parameter_normalizer is None:
            metrics = [scan.metric for scan in self.scans]
            self.metric_normalizer = Normalize(min(metrics),
                                               max(metrics),
                                               clip=True)
        else:
            self.metric_normalizer = self.cell_parameter_normalizer
        # Now plot the map
        return self.plot_map(*args, **kwargs)

    def set_metric_fwhm(self, phase_idx=0, *args, **kwargs):
        for scan in display_progress(self.scans, 'Culculating peak widths'):
            phase = scan.phases[phase_idx]
            scan.metric = scan.refinement.fwhm(phase=phase)

    def prepare_mapping_data(self):
        self.refine_scans()
        return super().prepare_mapping_data()

    def refine_scans(self):
        """
        Refine a series of parameters on each scan. Continue if an
        exceptions.RefinementError occurs.
        """
        for scan in display_progress(self.scans, 'Reticulating splines'):
            try:
                current_step = 'background'
                scan.refinement.refine_background()
                current_step = 'displacement'
                scan.refinement.refine_displacement()
                current_step = 'peak_widths'
                scan.refinement.refine_peak_widths()
                current_step = 'unit cells'
                scan.refinement.refine_unit_cells()
                current_step = 'scale factors'
                scan.refinement.refine_scale_factors()
            except exceptions.SingularMatrixError as e:
                # Display an error message on exception and then coninue fitting
                msg = "{coords}: {msg}".format(coords=scan.cube_coords, msg=e)
                print(msg)
            except exceptions.DivergenceError as e:
                msg = "{coords}: DivergenceError while refining {step}".format(
                    coords=scan.cube_coords,
                    step=current_step
                )
                print(msg)
