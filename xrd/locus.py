# -*- coding: utf-8 --*
import os

from mapping.locus import Locus
from xrd.scan import XRDScan
from refinement.native import NativeRefinement


class XRDLocus(Locus):
    diffractogram_is_loaded = False

    def __init__(self, *args, phases=[], background_phases=[],
                 two_theta_range=(10, 80),
                 refinement=NativeRefinement, **kwargs):
        ret = super().__init__(*args, **kwargs)
        # Attach XRDScan object
        self.xrdscan = XRDScan(phases=phases,
                               background_phases=background_phases,
                               refinement=refinement,
                               two_theta_range=two_theta_range,)
        # Filename is not passed to XRDScan constructor to delay
        # loading the datafile
        self.xrdscan.filename = self.filename
        self.xrdscan.refinement.basename = os.path.splitext(self.filename)[0]
        self.load_diffractogram(self.filename)
        return ret

    @property
    def diffractogram(self):
        return self.xrdscan.diffractogram

    @diffractogram.setter
    def diffractogram(self, newDf):
        self.xrdscan.diffractogram = newDf

    @property
    def refinement(self):
        return self.xrdscan.refinement

    def load_diffractogram(self, filename=None):
        if filename is None:
            filename = self.filename
        # Checking for existence of file allows for partial maps
        if filename is not None and os.path.isfile(filename):
            df = self.xrdscan.load_diffractogram(filename)
        else:
            df = None
        return df

    @property
    def phases(self):
        return self.xrdscan.phases

    @property
    def signal_level(self):
        """Intensity of all the phases"""
        return sum([p.scale_factor for p in self.phases])

    @property
    def reliability(self):
        """Normalized signal intensity from 0 to 1."""
        normalizer = self.parent_map.reliability_normalizer
        reliability = normalizer(self.signal_level)
        refinement_confidence = self.xrdscan.refinement.confidence()
        return reliability * refinement_confidence

    @property
    def metric_details(self):
        """Returns a string describing how the metric was calculated."""
        return self.refinement.details()

    def plot_diffractogram(self, ax=None):
        return self.xrdscan.plot_diffractogram(ax=ax)

    def axes_title(self):
        """Determine diffractogram axes title from cube coordinates."""
        title = 'XRD Diffractogram at ({i}, {j}, {k})'.format(
            i=self.cube_coords[0],
            j=self.cube_coords[1],
            k=self.cube_coords[2],
        )
        return title

    def refine_background(self):
        return self.xrdscan.refinement.refine_background()

    def refine_scale_factors(self):
        return self.xrdscan.refinement.refine_scale_factors()

    def refine_unit_cells(self):
        return self.xrdscan.refinement.refine_unit_cells()

    @property
    def data_dict(self):
        # Get general locus data
        dataDict = super().data_dict
        # Save data specific to X-ray diffraction
        dataDict['diffractogram'] = self.diffractogram
        dataDict['refinement'] = self.refinement.data_dict
        dataDict['phases'] = [phase.data_dict for phase in self.phases]
        return dataDict

    def restore_data_dict(self, new_data):
        self.diffractogram_is_loaded = new_data['diffractogram'] is not None
        self.diffractogram = new_data.pop('diffractogram')
        self.refinement.data_dict = new_data.pop('refinement')
        # Load phases
        phases = new_data.pop('phases')
        for idx, phase in enumerate(self.phases):
            phase.data_dict = phases[idx]
        super().restore_data_dict(new_data)
