# -*- coding: utf-8 -*-

import os

from gi.repository import Gtk, Gdk
from matplotlib import figure
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas

from mapping.coordinates import Cube

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
        image = '{0}/../images/icon.png'.format(directory)
        self.set_icon_from_file(image)
        self.set_default_size(1000, 1000)
        # Prepare layout box
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.add(box)
        # Prepare numerical text summary area
        self.dataSummary = DataSummaryBox(orientation=Gtk.Orientation.VERTICAL)
        box.pack_start(self.dataSummary, False, False, 10)
        self.dataSummary.set_default_data()
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
        self.draw_plots()
        # Connect to keypress event for changing position
        self.connect('key_press_event', self.on_key_press)
        # Connect to mouse click event
        fig.canvas.mpl_connect('button_press_event', self.click_callback)
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
            self.xrd_map.plot_diffractogram(ax=self.diffractogramAxes)
        # Draw individual scan's image or histogram
        self.scanImageAxes.clear()
        if activeScan:
            activeScan.plot_image(ax=self.scanImageAxes)
        else:
            self.xrd_map.plot_histogram(ax=self.scanImageAxes)
            self.scanImageAxes.set_aspect('auto')
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
        self.update_details()

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
        self.update_details()

    def update_details(self):
        """Set the sidebar text details."""
        if self.local_mode == True:
            self.dataSummary.update_data(self.currentScan)
        else:
            self.dataSummary.set_default_data()

    def main(self):
        Gtk.main()


class LeftLabel(Gtk.Label):
    """Label with text left aligned."""
    def __init__(self, *args, **kwargs):
        kwargs['xalign'] = 0
        return super(LeftLabel, self).__init__(*args, **kwargs)


class DetailBox(Gtk.Box):
    def __init__(self, *args, heading=None, **kwargs):
        kwargs['orientation'] = Gtk.Orientation.VERTICAL
        retVal = super(DetailBox, self).__init__(*args, **kwargs)
        # Create section heading
        self.headingLabel = Gtk.Label(xalign=0)
        markup = '<b><big>{text}</big></b>'.format(text=heading)
        self.headingLabel.set_markup(markup)
        self.pack_start(self.headingLabel, False, False, 0)
        # Prepare labels for populating later
        self.prepare_labels()
        return retVal


class LocationBox(DetailBox):
    def prepare_labels(self):
        # Label for XY coords
        box = Gtk.Box()
        self.pack_start(box, False, False, 0)
        box.pack_start(LeftLabel("XY: "), False, False, 0)
        self.xyLabel = LeftLabel("0")
        box.pack_start(self.xyLabel, False, False, 0)
        # Label for Cube coords
        box = Gtk.Box()
        self.pack_start(box, False, False, 0)
        box.pack_start(LeftLabel("Cube: "), False, False, 0)
        self.cubeLabel = LeftLabel("0")
        box.pack_start(self.cubeLabel, False, False, 0)

    def update_labels(self, scan):
        xyCoords = scan.xy_coords()
        xyStr = "({x:.02f}, {y:0.2f})".format(x=xyCoords[0], y=xyCoords[1])
        self.xyLabel.set_text(xyStr)
        self.cubeLabel.set_text(str(scan.cube_coords))

    def set_default_labels(self):
        self.xyLabel.set_text("N/A")
        self.cubeLabel.set_text("N/A")


class ValueBox(DetailBox):
    """Box shows a raw and normalized value, plus a space for other notes."""
    def prepare_labels(self):
        # Label for raw value
        box = Gtk.Box()
        self.pack_start(box, False, False, 0)
        box.pack_start(LeftLabel("Raw: "), False, False, 0)
        self.rawLabel = LeftLabel("0")
        box.pack_start(self.rawLabel, False, False, 0)
        # Label for normalized value
        box = Gtk.Box()
        self.pack_start(box, False, False, 0)
        box.pack_start(LeftLabel("Normalized: "), False, False, 0)
        self.normLabel = LeftLabel()
        box.pack_start(self.normLabel, False, False, 0)
        # Label for additional info
        self.otherLabel = LeftLabel()
        self.otherLabel.set_line_wrap(True)
        self.pack_start(self.otherLabel, False, False, 0)

    def set_default_labels(self):
        # Set default values
        self.rawLabel.set_text("N/A")
        self.normLabel.set_text("N/A")
        self.otherLabel.hide()


class MetricBox(ValueBox):
    def update_labels(self, scan):
        # Set values from scan
        self.rawLabel.set_text("{:.03f}".format(scan.metric))
        self.normLabel.set_text("{:.03f}".format(scan.metric_normalized))
        self.otherLabel.set_text(scan.metric_details)
        self.otherLabel.show()


class ReliabilityBox(ValueBox):
    def update_labels(self, scan):
        # Set values from scan
        self.rawLabel.set_text("{:.03f}".format(scan.reliability_raw))
        self.normLabel.set_text("{:.03f}".format(scan.reliability))


class DataSummaryBox(Gtk.Box):
    """Three-section box that shows a summary of data for a Scan."""
    padding = 10
    def __init__(self, *args, **kwargs):
        retVal = super(DataSummaryBox, self).__init__(*args, **kwargs)
        # Prepare Location box
        self.locBox = LocationBox(heading="Location")
        self.pack_start(self.locBox, False, False, self.padding)
        # Prepare Metric box
        self.metricBox = MetricBox(heading="Metric")
        self.pack_start(self.metricBox, False, False, self.padding)
        # Prepare Reliability box
        self.reliabilityBox = ReliabilityBox(heading="Reliability")
        self.pack_start(self.reliabilityBox, False, False, self.padding)
        return retVal

    def update_data(self, scan):
        self.locBox.update_labels(scan=scan)
        self.metricBox.update_labels(scan=scan)
        self.reliabilityBox.update_labels(scan=scan)

    def set_default_data(self):
        self.locBox.set_default_labels()
        self.metricBox.set_default_labels()
        self.reliabilityBox.set_default_labels()
