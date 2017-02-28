# -*- coding: utf-8 -*-

import math
import os
import pickle
import warnings

from matplotlib import pyplot, cm, patches, colors
import numpy as np
import scipy

from .coordinates import Cube
# from .locus import Locus, DummyLocus
# from .colormaps import cmaps
from .plots import new_axes, dual_axes, set_outside_ticks
from .utilities import prog, xycoord


# def normalizer(data, norm_range):
#     """Factory for creating normalizers for some data to the range
#     given. If norm_range is None, this will norm the data to cover the
#     full range. Returns Normalizer object that is callable on new data.
#     """
#     if norm_range is None:
#         norm = colors.Normalize(min(data), max(data), clip=True)
#     else:
#         norm = colors.Normalize(min(norm_range), max(norm_range), clip=True)
#     return norm



# class DummyMap(Map):
#     """
#     Sample that returns a dummy map for testing.
#     """

#     def composite_image(self):
#         # Stub image to show for layout purposes
#         directory = os.path.dirname(os.path.realpath(__file__))
#         # Read a cached composite image from disk
#         image = scipy.misc.imread(
#             '{0}/../images/test-composite-image.png'.format(directory)
#         )
#         return image

#     def mapscan_metric(self, scan):
#         # Just return the distance from bottom left to top right
#         p = scan.cube_coords[0]
#         rows = scan.xrd_map.rows
#         r = (p / 2 / rows) + 0.5
#         return r

#     def plot_map(self, *args, **kwargs):
#         # Ensure that "diffractogram is loaded" for each scan
#         for locus in self.loci:
#             locus.diffractogram_is_loaded = True
#             p = locus.cube_coords[0]
#             rows = locus.parent_map.rows
#             r = (p / 2 / rows) + 0.5
#             locus.metric = r
#         return super().plot_map(*args, **kwargs)

#     def create_loci(self):
#         """Populate the loci array with new scans in a hexagonal array."""
#         raise NotImplementedError("Use gadds._path()")




# class IORMap(Map):
#     """One-off material for submitting an image of the Image of Research
#     competition at UIC."""
#     metric_normalizer = colors.Normalize(0, 1, clip=True)
#     reliability_normalizer = colors.Normalize(2.3, 4.5, clip=True)
#     charged_peak = '331'
#     discharged_peak = '400'
#     reliability_peak = '400'

#     def mapscan_metric(self, scan):
#         area = self.peak_area(scan, self.peak_list[self.charged_peak])
#         return area
