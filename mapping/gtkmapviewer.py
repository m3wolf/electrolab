# -*- coding: utf-8 -*-

import os

import numpy as np
import gi
from gi.repository import Gtk, Gdk
from matplotlib import figure
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas

# noqa

from mapping.coordinates import Cube
from txm.gtk_viewer import WatchCursor
from utilities import xycoord

gi.require_version('Gtk', '3.0')

class GtkMapViewer():
    """
    A set of plots for interactive data analysis.
    """
    local_mode = False
    current_locus = 0
    beam = None
    image_hexagon = None
    composite_hexagon = None
    active_metric = 'integral'

    def __init__(self, *args, parent_map, **kwargs):
        self.parent_map = parent_map
        # Build GTK window
        self.builder = Gtk.Builder()
        # Load the GUI from a glade file
        gladefile = os.path.join(os.path.dirname(__file__),
                                 "map_viewer.glade")
        self.builder.add_from_file(gladefile)
        self.window = self.builder.get_object('MapViewerWindow')
        # Load icon
        directory = os.path.dirname(os.path.realpath(__file__))
        image = '{0}/../images/icon.png'.format(directory)
        self.window.set_icon_from_file(image)
        self.window.set_default_size(1000, 1000)
        # Populate the combobox with list of available metrics
        self.metric_combo = self.builder.get_object('MetricComboBox')
        self.metric_list = Gtk.ListStore(str, str)
        choices = self.parent_map.valid_metrics()
        for choice in choices:
            uppercase = " ".join(
                [word.capitalize() for word in choice.split('_')]
            )
            metric_iter = self.metric_list.append([uppercase, choice])
            # Save active group for later initialization
            # if rep == self.frameset.default_representation:
            #     active_rep = rep_iter
        self.metric_combo.set_model(self.metric_list)
        self.metric_combo.set_active_iter(metric_iter)
        # if active_rep:
        #     self.metric_combo.set_active_iter(active_rep)

        # Prepare mapping figures
        fig = figure.Figure(figsize=(13.8, 10))
        self.fig = fig
        fig.figurePatch.set_facecolor('white')
        sw = self.builder.get_object('MapPanel')
        # Prepare plotting area
        canvas = FigureCanvas(self.fig)
        canvas.set_size_request(400, 400)
        sw.add(canvas)
        self.draw_plots()
        # Connect to keypress event for changing position
        self.window.connect('key_press_event', self.on_key_press)
        # Connect to mouse click event
        fig.canvas.mpl_connect('button_press_event', self.click_callback)
        # Connect other event handlers
        handlers = {
            'change-metric': WatchCursor(self.change_metric,
                                         windows=[self.window]),
        }
        self.builder.connect_signals(handlers)
        self.window.connect('delete-event', Gtk.main_quit)
        # Set initial text values
        self.update_details()

    def show(self):
        self.window.show_all()
        Gtk.main()

    def draw_plots(self, locus=None):
        """
        (re)draw the plots on the gtk window
        """
        xrdMap = self.parent_map
        self.fig.clear()
        # Prepare plots
        self.mapAxes = self.fig.add_subplot(221)
        xrdMap.plot_map(ax=self.mapAxes, metric=self.active_metric)
        self.mapAxes.set_aspect(1)
        self.compositeImageAxes = self.fig.add_subplot(223)
        xrdMap.plot_composite_image(ax=self.compositeImageAxes)
        self.locusImageAxes = self.fig.add_subplot(224)
        self.update_plots()

    def update_plots(self):
        """Respond to changes in the selected locus."""
        # Clear old highlights
        try:
            self.beam.remove()
        except (ValueError, AttributeError):
            pass
        finally:
            self.beam = None
            # self.composite_hexagon.remove()
            # self.composite_hexagon = None
            # self.image_hexagon.remove()
            # self.image_hexagon = None
        # Check if a locus should be highlighted
        if self.local_mode:
            activeLocus = self.current_locus
        else:
            activeLocus = None
        # Plot diffractogram (either bulk or local)
        self.locusAxes = self.fig.add_subplot(222)
        self.locusAxes.cla()
        self.plot_locus_detail(locus=activeLocus)
        # Draw individual locus's image or histogram
        self.locusImageAxes.clear()
        print("Todo: re-implement the local image plotting")
        #     activeLocus.plot_image(ax=self.locusImageAxes)
        # else:
        #     self.parent_map.plot_histogram(ax=self.locusImageAxes)
        #     self.locusImageAxes.set_aspect('auto')
        self.parent_map.plot_histogram(ax=self.locusImageAxes,
                                       metric=self.active_metric)
        self.locusImageAxes.set_aspect('auto')
        # Highlight the hexagon on the map and composite image
        if activeLocus is not None:
            # self.map_hexagon = activeLocus.highlight_beam(ax=self.mapAxes)
            self.beam = self.parent_map.highlight_beam(ax=self.mapAxes, locus=activeLocus)
            # self.composite_hexagon = activeLocus.highlight_beam(
            #     ax=self.compositeImageAxes)
            # self.image_hexagon = activeLocus.highlight_beam(
            #     ax=self.locusImageAxes)
            # self.mapAxes.draw_artist(self.map_hexagon)
        # Force a redraw of the canvas since Gtk won't do it
        self.fig.canvas.draw()

    def plot_locus_detail(self, locus):
        self.parent_map.plot_diffractogram(ax=self.locusAxes, index=locus)
        # Return some random data
        # twoTheta = numpy.linspace(10, 80, num=700)
        # counts = numpy.random.rand(len(twoTheta))
        # self.locusAxes.plot(twoTheta, counts)
        # return self.locusAxes

    def on_key_press(self, widget, event, user_data=None):
        oldCoords = np.array(self.parent_map.xy_by_locus(self.current_locus))
        # oldCoords = self.current_locus.cube_coords
        newCoords = oldCoords
        # Check for arrow keys -> move to new location on map
        if not self.local_mode:
            self.local_mode = True
        elif event.keyval == Gdk.KEY_Left:
            newCoords = oldCoords + np.array((-1, 0))
        elif event.keyval == Gdk.KEY_Right:
            newCoords = oldCoords + np.array((1, 0))
        elif event.keyval == Gdk.KEY_Up:
            newCoords = oldCoords + np.array((0, 1))
        elif event.keyval == Gdk.KEY_Down:
            newCoords = oldCoords + np.array((0, -1))
        elif event.keyval == Gdk.KEY_Escape:
            # Return to bulk view
            self.local_mode = False
        # Check if new coordinates are valid and update locs
        locus = self.parent_map.locus_by_xy(xy=newCoords)
        if locus:
            self.current_locus = locus
        self.update_plots()
        self.update_details()

    def click_callback(self, event):
        """Detect and then update which locus is active."""
        inMapAxes = event.inaxes == self.mapAxes
        inCompositeAxes = event.inaxes == self.compositeImageAxes
        inImageAxes = event.inaxes == self.locusImageAxes
        if (inMapAxes or inCompositeAxes or inImageAxes):
            # Switch to new position on map
            locus = self.parent_map.locus_by_xy((event.xdata, event.ydata))
            self.local_mode = True
            self.current_locus = locus
        else:
            # Reset local_mode
            self.local_mode = False
        self.update_plots()
        self.update_details()

    def change_metric(self, widget, object=None):
        new_metric = self.metric_list[widget.get_active_iter()][1]
        self.active_metric = new_metric
        self.draw_plots()

    def update_details(self):
        """Set the sidebar text details."""
        xylabel = self.builder.get_object('XYLabel')
        cubelabel = self.builder.get_object('CubeLabel')
        indexlabel = self.builder.get_object('IndexLabel')
        metric_raw_label = self.builder.get_object('MetricRawLabel')
        metric_norm_label = self.builder.get_object('MetricNormLabel')
        metric_method = self.builder.get_object('RefinementLabel')
        reliability_raw_label = self.builder.get_object('ReliabilityRawLabel')
        reliability_norm_label = self.builder.get_object('ReliabilityNormLabel')
        if self.local_mode:
            locus = self.current_locus
            # Set location labels
            xy = self.parent_map.xy_by_locus(locus=locus)
            xyStr = "({x:.02f}, {y:0.2f})".format(x=xy.x, y=xy.y)
            xylabel.set_text(xyStr)
            cubelabel.set_text(xyStr)
            indexlabel.set_text(str(self.current_locus))
            # Set metric labels
            metric = self.parent_map.metric(param=self.active_metric, locus=locus)[0]
            metric_raw_label.set_text("{:.03f}".format(metric))
            # metric_norm_label.set_text("{:.03f}".format(locus.metric_normalized))
            # metric_method.set_text(locus.metric_details)
            # Set reliability labels
            # reliability_raw_label.set_text("{:.03f}".format(locus.signal_level))
            # reliability_norm_label.set_text("{:.03f}".format(locus.reliability))
        else:
            # self.dataSummary.set_default_data()
            default_text = "N/A"
            xylabel.set_text(default_text)
            cubelabel.set_text(default_text)
            indexlabel.set_text(default_text)
            metric_raw_label.set_text(default_text)
            metric_norm_label.set_text(default_text)
            metric_method.set_text("Details unavailable")
            reliability_raw_label.set_text(default_text)
            reliability_norm_label.set_text(default_text)


# class LeftLabel(Gtk.Label):
#     """Label with text left aligned."""
#     def __init__(self, *args, **kwargs):
#         kwargs['xalign'] = 0
#         return super(LeftLabel, self).__init__(*args, **kwargs)


# class DetailBox(Gtk.Box):
#     def __init__(self, *args, heading=None, **kwargs):
#         kwargs['orientation'] = Gtk.Orientation.VERTICAL
#         retVal = super(DetailBox, self).__init__(*args, **kwargs)
#         # Create section heading
#         self.headingLabel = Gtk.Label(xalign=0)
#         markup = '<b><big>{text}</big></b>'.format(text=heading)
#         self.headingLabel.set_markup(markup)
#         self.pack_start(self.headingLabel, False, False, 0)
#         # Prepare labels for populating later
#         self.prepare_labels()
#         return retVal


# class LocationBox(DetailBox):
#     def prepare_labels(self):
#         # Label for XY coords
#         box = Gtk.Box()
#         self.pack_start(box, expand=False, fill=False, padding=0)
#         box.pack_start(child=LeftLabel("XY: "),
#                        expand=False,
#                        fill=False,
#                        padding=0)
#         self.xyLabel = LeftLabel("0")
#         box.pack_start(self.xyLabel, expand=False, fill=False, padding=0)
#         # Label for Cube coords
#         box = Gtk.Box()
#         self.pack_start(box, expand=False, fill=False, padding=0)
#         box.pack_start(LeftLabel("Cube: "), expand=False, fill=False, padding=0)
#         self.cubeLabel = LeftLabel("0")
#         box.pack_start(self.cubeLabel, expand=False, fill=False, padding=0)

#     def update_labels(self, locus):
#         xyCoords = locus.xy_coords()
#         xyStr = "({x:.02f}, {y:0.2f})".format(x=xyCoords[0], y=xyCoords[1])
#         self.xyLabel.set_text(xyStr)
#         self.cubeLabel.set_text(str(locus.cube_coords))

#     def set_default_labels(self):
#         self.xyLabel.set_text("N/A")
#         self.cubeLabel.set_text("N/A")


# class ValueBox(DetailBox):
#     """Box shows a raw and normalized value, plus a space for other notes."""
#     def prepare_labels(self):
#         # Label for raw value
#         box = Gtk.Box()
#         self.pack_start(box, False, False, 0)
#         box.pack_start(LeftLabel("Raw: "), False, False, 0)
#         self.rawLabel = LeftLabel("0")
#         box.pack_start(self.rawLabel, False, False, 0)
#         # Label for normalized value
#         box = Gtk.Box()
#         self.pack_start(box, False, False, 0)
#         box.pack_start(LeftLabel("Normalized: "), False, False, 0)
#         self.normLabel = LeftLabel()
#         box.pack_start(self.normLabel, False, False, 0)
#         # Label for additional info
#         self.otherLabel = LeftLabel()
#         self.otherLabel.set_line_wrap(True)
#         self.pack_start(self.otherLabel, False, False, 0)

#     def set_default_labels(self):
#         # Set default values
#         self.rawLabel.set_text("N/A")
#         self.normLabel.set_text("N/A")
#         self.otherLabel.hide()


# class MetricBox(ValueBox):
#     def update_labels(self, locus):
#         # Set values from locus
#         self.rawLabel.set_text("{:.03f}".format(locus.metric))
#         self.normLabel.set_text("{:.03f}".format(locus.metric_normalized))
#         self.otherLabel.set_text(locus.metric_details)
#         self.otherLabel.show()


# class ReliabilityBox(ValueBox):
#     def update_labels(self, locus):
#         # Set values from locus
#         self.rawLabel.set_text("{:.03f}".format(locus.signal_level))
#         self.normLabel.set_text("{:.03f}".format(locus.reliability))


# class DataSummaryBox(Gtk.Box):
#     """Three-section box that shows a summary of data for a Locus."""
#     padding = 10

#     def __init__(self, *args, **kwargs):
#         retVal = super(DataSummaryBox, self).__init__(*args, **kwargs)
#         # Prepare Location box
#         self.locBox = LocationBox(heading="Location")
#         self.pack_start(self.locBox, False, False, self.padding)
#         # Prepare Metric box
#         self.metricBox = MetricBox(heading="Metric")
#         self.pack_start(self.metricBox, False, False, self.padding)
#         # Prepare Reliability box
#         self.reliabilityBox = ReliabilityBox(heading="Reliability")
#         self.pack_start(self.reliabilityBox, False, False, self.padding)
#         return retVal

#     def update_data(self, locus):
#         self.locBox.update_labels(locus=locus)
#         self.metricBox.update_labels(locus=locus)
#         self.reliabilityBox.update_labels(locus=locus)

#     def set_default_data(self):
#         self.locBox.set_default_labels()
#         self.metricBox.set_default_labels()
#         self.reliabilityBox.set_default_labels()
