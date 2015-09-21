# -*- coding: utf-8 --*
import os

from mapping.locus import Locus
from xrd.scan import XRDScan
from refinement.native import NativeRefinement

class XRDLocus(Locus):

    def __init__(self, *args, phases=[], background_phases=[],
                 two_theta_range=(10, 80), refinement=NativeRefinement, **kwargs):
        ret = super().__init__(*args, **kwargs)
        # Attach XRDScan object
        self.xrdscan = XRDScan(phases=phases)
        return ret


    @property
    def diffractogram(self):
        return self.xrdscan.diffractogram

    @diffractogram.setter
    def diffractogram(self, newDf):
        self.xrdscan.diffractogram = newDf

    def load_diffractogram(self, filename):
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

    @reliability.setter
    def reliability(self, value):
        pass

    def refine_background(self):
        return self.xrdscan.refinement.refine_background()

    def refine_scale_factors(self):
        return self.xrdscan.refinement.refine_scale_factors()

    def refine_unit_cells(self):
        return self.xrdscan.refinement.refine_unit_cells()
