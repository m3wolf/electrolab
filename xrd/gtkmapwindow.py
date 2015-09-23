# -*- coding: utf-8 -*-

from mapping.gtkmapwindow import GtkMapWindow

class GtkXrdMapWindow(GtkMapWindow):

    def plot_locus_detail(self, locus):
        if locus:
            locus.plot_diffractogram(ax=self.locusAxes)
        else:
            self.parent_map.plot_diffractogram(ax=self.locusAxes)
        return self.locusAxes
