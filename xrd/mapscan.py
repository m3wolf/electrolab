# -*- coding: utf-8 --*

from mapping.mapscan import MapScan, cached_property

class XRDMapScan(MapScan):
    @property
    def signal_level(self):
        """Intensity of all the phases"""
        return sum([p.scale_factor for p in self.phases])

    # @cached_property
    @property
    def reliability(self):
        """Normalized signal intensity from 0 to 1."""
        normalizer = self.xrd_map.reliability_normalizer
        reliability = normalizer(self.signal_level)
        refinement_confidence = self.refinement.confidence()
        return reliability * refinement_confidence

    @reliability.setter
    def reliability(self, value):
        pass
