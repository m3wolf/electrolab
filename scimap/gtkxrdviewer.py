# -*- coding: utf-8 -*-

import warnings

from ..mapping.gtkmapviewer import GtkMapViewer


class GtkXrdMapViewer(GtkMapViewer):

    def __init__(self, *args, **kwargs):
        warnings.warn(DeprecationWarning(), "Just use the regular GtkMapViewer")
        super().__init__(*args, **kwargs)

    # def plot_locus_detail(self, locus):
    #     if locus:
    #         locus.plot_diffractogram(ax=self.locusAxes)
    #     else:
    #         self.parent_map.plot_diffractogram(ax=self.locusAxes)
    #     return self.locusAxes
