# -*- coding: utf-8 -*-

import os

from gi.repository import Gtk, Gdk
from matplotlib import figure
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas

from coordinates import Cube

class GtkMapWindow(Gtk.Window):
    """
    A set of plots for interactive data analysis.
    """
    local_mode = False
    map_hexagon = None
    image_hexagon = None
    composite_hexagon = None
    def __init__(self, xrd_map, *args, **kwargs):
        self.xrd_map = xrd_map
        self.currentScan = self.xrd_map.scan(Cube(0, 0, 0))
        return_val = super(GtkMapWindow, self).__init__(*args, **kwargs)
        self.connect('delete-event', Gtk.main_quit)
        # Load icon
        directory = os.path.dirname(os.path.realpath(__file__))
        image = '{0}/images/icon.png'.format(directory)
        self.set_icon_from_file(image)
        self.set_default_size(1000, 1000)
        # Prepare layout box
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(box)
        # Prepare numerical text summary area
        label = Gtk.Label('hello, world')
        box.pack_start(label, False, False, 0)
        # Set up the matplotlib features
        fig = figure.Figure(figsize=(13.8, 10))
        self.fig = fig
        fig.figurePatch.set_facecolor('white')
        sw = Gtk.ScrolledWindow()
        # Prepare plotting area
        canvas = FigureCanvas(self.fig)
        canvas.set_size_request(400,400)
        box.pack_start(sw, True, True, 0)
        sw.add(canvas)
        # sw.add_with_viewport(canvas)
        self.draw_plots()
        # Connect to keypress event for changing position
        self.connect('key_press_event', self.on_key_press)
        # Connect to mouse click event
        fig.canvas.mpl_connect('button_press_event', self.click_callback)
        # canvas.add(label)
        return return_val

    def draw_plots(self, scan=None):
        """
        (re)draw the plots on the gtk window
        """
        xrdMap = self.xrd_map
        self.fig.clear()
        # Prepare plots
        self.mapAxes = self.fig.add_subplot(221)
        xrdMap.plot_map(ax=self.mapAxes)
        self.mapAxes.set_aspect(1)
        self.compositeImageAxes = self.fig.add_subplot(223)
        xrdMap.plot_composite_image(ax=self.compositeImageAxes)
        self.scanImageAxes = self.fig.add_subplot(224)
        self.update_plots()

    def update_plots(self):
        """Respond to changes in the selected scan."""
        # Clear old highlights
        if self.map_hexagon:
            self.map_hexagon.remove()
            self.map_hexagon = None
            self.composite_hexagon.remove()
            self.composite_hexagon = None
            self.image_hexagon.remove()
            self.image_hexagon = None
        # Check if a scan should be highlighted
        if self.local_mode:
            activeScan = self.currentScan
        else:
            activeScan = None
        # Plot diffractogram (either bulk or local)
        self.diffractogramAxes = self.fig.add_subplot(222)
        self.diffractogramAxes.cla() # Clear axes
        if activeScan:
            activeScan.plot_diffractogram(ax=self.diffractogramAxes)
        else:
            self.xrd_map.plot_bulk_diffractogram(ax=self.diffractogramAxes)
        # Draw individual scan's image or histogram
        self.scanImageAxes.cla()
        if activeScan:
            activeScan.plot_image(ax=self.scanImageAxes)
        else:
            self.xrd_map.plot_histogram(ax=self.scanImageAxes)
        # Highlight the hexagon on the map and composite image
        if activeScan:
            self.map_hexagon = activeScan.highlight_beam(ax=self.mapAxes)
            self.composite_hexagon = activeScan.highlight_beam(
                ax=self.compositeImageAxes)
            self.image_hexagon = activeScan.highlight_beam(
                ax=self.scanImageAxes)
            self.mapAxes.draw_artist(self.map_hexagon)
        # Force a redraw of the canvas since Gtk won't do it
        self.fig.canvas.draw()

    def on_key_press(self, widget, event, user_data=None):
        oldCoords = self.currentScan.cube_coords
        newCoords = oldCoords
        # Check for arrow keys -> move to new location on map
        if not self.local_mode:
            self.local_mode = True
        elif event.keyval == Gdk.KEY_Left:
            newCoords = oldCoords + Cube(0, 1, -1)
        elif event.keyval == Gdk.KEY_Right:
            newCoords = oldCoords + Cube(0, -1, 1)
        elif event.keyval == Gdk.KEY_Up:
            newCoords = oldCoords + Cube(1, 0, -1)
        elif event.keyval == Gdk.KEY_Down:
            newCoords = oldCoords + Cube(-1, 0, 1)
        elif event.keyval == Gdk.KEY_Escape:
            # Return to bulk view
            self.local_mode = False
        # Check if new coordinates are valid and update scan
        scan = self.xrd_map.scan(newCoords)
        if scan:
            self.currentScan = scan
        self.update_plots()

    def click_callback(self, event):
        """Detect and then update which scan is active."""
        inMapAxes = event.inaxes == self.mapAxes
        inCompositeAxes = event.inaxes == self.compositeImageAxes
        inImageAxes = event.inaxes == self.scanImageAxes
        if (inMapAxes or inCompositeAxes or inImageAxes):
            # Switch to new position on map
            scan = self.xrd_map.scan_by_xy((event.xdata, event.ydata))
            if not self.local_mode:
                self.local_mode = True
            elif scan:
                self.currentScan = scan
        else:
            # Reset local_mode
            self.local_mode = False
        self.update_plots()

    def main(self):
        Gtk.main()
