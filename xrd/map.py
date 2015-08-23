# -*- coding: utf-8 --*

import exceptions
from mapping.map import Map, display_progress
from xrd.mapscan import XRDMapScan

class XRDMap(Map):
    cell_class = XRDMapScan
    def set_metric_phase_ratio(self, phase_idx=0):
        """Set the plotting metric as the proportion of given phase."""
        for scan in display_progress(self.scans, 'Calculating metrics'):
            phase_scale = scan.phases[phase_idx].scale_factor
            total_scale = sum([phase.scale_factor for phase in scan.phases])
            scan.metric = phase_scale/total_scale

    def plot_phase_ratio(self, phase_idx=0, *args, **kwargs):
        """Plot a map of the ratio of the given phase index to all the phases"""
        self.set_metric_phase_ratio(phase_idx=0)
        return self.plot_map(*args, **kwargs)

    def set_metric_cell_parameter(self, parameter='a', phase_idx=0):
        for scan in display_progress(self.scans, 'Calculating cell parameters'):
            phase = scan.phases[phase_idx]
            scan.metric = getattr(phase.unit_cell, parameter)

    def plot_cell_parameter(self, parameter='a', phase_idx=0, *args, **kwargs):
        self.set_metric_cell_parameter(parameter, phase_idx)
        # Now plot the map
        return self.plot_map(*args, **kwargs)

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
                scan.refinement.refine_background()
                scan.refinement.refine_displacement()
                scan.refinement.refine_unit_cells()
                scan.refinement.refine_scale_factors()
            except exceptions.SingularMatrixError as e:
                # Display an error message on exception and then coninue fitting
                msg = "{coords}: {msg}".format(coords=scan.cube_coords, msg=e)
                print(msg)
