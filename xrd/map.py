# -*- coding: utf-8 --*

from mapping.map import Map, display_progress

class XRDMap(Map):

    def plot_phase_ratio(self, *args, phase_idx=0, **kwargs):
        """Plot a map of the ratio of the given phase index to all the phases"""
        for scan in display_progress(self.scans, 'Calculating metrics'):
            phase_scale = scan.phases[phase_idx].scale_factor
            total_scale = sum([phase.scale_factor for phase in scan.phases])
            scan.metric = phase_scale/total_scale
        # Calculate reliabilities
        self.calculate_reliabilities()
        # Now plot the map
        return self.plot_map(*args, **kwargs)

    def calculate_reliabilities(self, normalized_range=None):
        """Un-reliable cells are mapped with lower opacity."""
        if normalized_range is None:
            normalizer = self.reliability_normalizer
        else:
            normalizer = Normalize(*normalized_range, clip=True)
        # Calculate reliabilities for each scan based on phase intensitieis
        for scan in display_progress(self.scans, 'Reticulating splines'):
            signal = sum([phase.scale_factor for phase in scan.phases])
            scan.reliability = normalizer(signal)

    def prepare_mapping_data(self):
        self.refine_scans()
        return super().prepare_mapping_data()

    def refine_scans(self):
        """
        Refine a series of parameters on each scan. Continue if an
        exceptions.RefinementError occurs.
        """
        for scan in display_progress(self.scans, 'Decomposing patterns'):
            try:
                scan.refinement.refine_background()
                scan.refinement.refine_displacement()
                scan.refinement.refine_unit_cells()
                scan.refinement.refine_scale_factors()
            except exceptions.SingularMatrixError as e:
                # Display an error message on exception and then coninue fitting
                msg = "{coords}: {msg}".format(coords=scan.cube_coords, msg=e)
                print(msg)
