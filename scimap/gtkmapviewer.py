# -*- coding: utf-8 -*-

import os
import math
import collections
import threading

import numpy as np
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib
from matplotlib import figure
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas

# noqa
gi.require_version('Gtk', '3.0')

NormRange = collections.namedtuple('NormRange', ('min', 'max'))

WATCH_CURSOR = Gdk.Cursor(Gdk.CursorType.WATCH)
ARROW_CURSOR = Gdk.Cursor(Gdk.CursorType.ARROW)


class WatchCursor():
    """Factory returns a function that Perform some slow action `target`
    with a watch cursor over the given `windows`. When writing the
    `target` function, make sure to use GLib.idle_add for anything
    that modifies the UI.
    """
    _timer = None

    def __init__(self, target, windows, threading=True):
        self.target = target
        self.windows = windows
        self.threading = threading

    def __call__(self, *args, delay=0):
        """Start the watch cursor with a delay of `delay` in milliseconds."""
        def dostuff(*args):
            # Unpack user data
            windows, target, *more = args
            # Start watch cursor
            for window in windows:
                real_window = window.get_window()
                if real_window:
                    real_window.set_cursor(WATCH_CURSOR)
                    window.set_sensitive(False)
            # Call the actual function
            target(*more)
            # Remove watch cursor
            for window in windows:
                real_window = window.get_window()
                if real_window:
                    real_window.set_cursor(ARROW_CURSOR)
                    window.set_sensitive(True)
            self._timer = None
            return False

        # Start target process
        if self.threading:
            # Reset timer
            if self._timer is not None:
                GLib.source_remove(self._timer)
            self._timer = GLib.timeout_add(delay, dostuff,
                                           self.windows, self.target,
                                           *args)
        else:
            dostuff(self.windows, self.target, *args)


class GtkMapViewer():
    """A set of plots for interactive data analysis.

    Arguments
    ---------
    - parent_map : A Map() object that contains the data to be mapped.

    - metric : Parameter to map by default. Can be changed within the GUI.

    - metric_range : Starting range for the metric. Can be changed
      within the GUI. If omitted or None, the full range of metrics
      will be used.

    - alpha : Parameter to use for transparency by default. Can be
      changed within the GUI.

    - alpha_range : Starting range for transparency. Can be changed
      within the GUI. If omitted or None, the full range of values
      will be used.

    """
    local_mode = False
    current_locus = 0
    beam = None
    image_hexagon = None
    composite_hexagon = None
    _drawing_timer = None

    def __init__(self, parent_map, metric='a', metric_range=None,
                 alpha='integral', alpha_range=None, *args, **kwargs):
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
        image = '{0}/images/icon.png'.format(directory)
        self.window.set_icon_from_file(image)
        self.window.set_default_size(1000, 1000)
        # Populate the combobox with list of available metrics (and alphas)
        self.active_metric = metric
        self.active_alpha = alpha
        self.metric_combo = self.builder.get_object('MetricComboBox')
        self.alpha_combo = self.builder.get_object("AlphaComboBox")
        self.metric_list = Gtk.ListStore(str, str)
        choices = self.parent_map.valid_metrics()
        for choice in choices:
            uppercase = " ".join(
                [word.capitalize() for word in choice.split('_')]
            )
            metric_iter = self.metric_list.append([uppercase, choice])
            # Save active metric for later initialization
            if choice == self.active_metric:
                active_iter_metric = metric_iter
            if choice == self.active_alpha:
                active_iter_alpha = metric_iter
        self.metric_combo.set_model(self.metric_list)
        self.alpha_combo.set_model(self.metric_list)
        self.metric_combo.set_active_iter(active_iter_metric)
        self.alpha_combo.set_active_iter(active_iter_alpha)

        # Set ranges for metric and alpha spin buttons
        if metric_range is None:
            self.reset_metric_range()
        else:
            self.metric_range = metric_range
        if alpha_range is None:
            self.reset_alpha_range()
        else:
            self.alpha_range = alpha_range

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
        # handlers = {
        #     'change-metric': WatchCursor(self.change_metric,
        #                                  windows=[self.window]),
        #     'change-alpha': WatchCursor(self.change_alpha,
        #                                  windows=[self.window]),
        #     'change-norm-range': WatchCursor(self.change_norm_range,
        #                                      windows=[self.window]),
        # }
        # self.builder.connect_signals(handlers)
        self.builder.connect_signals(self)
        self.window.connect('delete-event', Gtk.main_quit)
        # Set initial text values
        self.update_details()

    def show(self):
        self.window.show_all()
        Gtk.main()

    def draw_plots(self, widget=None, object=None, delay=0):
        """(re)draw the plots on the gtk window. This takes place after a
        short delay to avoid excessive re-drawing (drawing can be
        slow).
        """
        # Set up an asynchronous function
        def _draw_plots():
            self.fig.clear()
            # Prepare plots
            self.mapAxes = self.fig.add_subplot(221)
            xrdMap = self.parent_map
            if self.show_alpha:
                alpha = self.active_alpha
            else:
                alpha = None
            xrdMap.plot_map(ax=self.mapAxes,
                            metric=self.active_metric,
                            metric_range=self.metric_range,
                            alpha=alpha,
                            alpha_range=self.alpha_range)
            self.mapAxes.set_aspect(1)
            self.compositeImageAxes = self.fig.add_subplot(223)
            xrdMap.plot_composite_image(ax=self.compositeImageAxes)
            self.locusImageAxes = self.fig.add_subplot(224)
            self.update_plots()

        # Prepare a timed watch cursor
        if getattr(self, '_draw_plots', None) is None:
            self._draw_plots = WatchCursor(_draw_plots,
                                           windows=[self.window],
                                           threading=True)
        self._draw_plots(delay=delay)

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
        # Plot the histogram
        if self.show_alpha:
            real_alpha = self.active_alpha
        else:
            real_alpha = None
        self.parent_map.plot_histogram(ax=self.locusImageAxes,
                                       metric=self.active_metric,
                                       metric_range=self.metric_range,
                                       weight=real_alpha,
                                       weight_range=self.alpha_range)
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
        self.reset_metric_range()
        self.draw_plots()

    def change_alpha(self, widget, object=None):
        new_alpha = self.metric_list[widget.get_active_iter()][1]
        self.active_alpha = new_alpha
        self.reset_alpha_range()
        self.draw_plots()

    def change_norm_range(self, widget, object=None):
        # Triggers the properties to adjust the boundaries automatically
        self.metric_range = self.metric_range
        self.alpha_range = self.alpha_range
        self.draw_plots(delay=500)

    def _set_adjustments(self, adjustments, value_range, spin_buttons):
        """Take a tuple of min and max GTK adjustments and a tuple of gtk spin
        buttons and set them to a tuple of values in
        `value_range`. Returns the step_size between the values.
        """
        adj_min, adj_max = adjustments
        btn_min, btn_max = spin_buttons
        val_min, val_max = value_range
        # Determine the adjustment step
        try:
            order_of_magnitude = round(math.log((val_max - val_min)/100, 10))
        except ValueError:
            order_of_magnitude = 0
            val_min = 0
            val_max = 1
        step = 10 ** order_of_magnitude
        # Freeze notifications to avoid race conditions
        f1, f2, f3, f4 = (adj_min.freeze_notify(),
                          adj_max.freeze_notify(),
                          adj_min.freeze_notify(),
                          adj_max.freeze_notify())
        with f1 as a, f2 as b, f3 as c, f4 as d:
            # Set minimum ranges
            if not val_min == adj_min.get_value():
                adj_min.set_upper(val_max)
                adj_min.set_lower(val_min - 1000 * step)
                adj_min.set_value(val_min)
            # Set maximum ranges
            if not val_max == adj_max.get_value():
                adj_max.set_lower(val_min)
                adj_max.set_upper(val_max + 1000 * step)
                adj_max.set_value(val_max)
            # Adjust precision of the buttons
            adj_min.set_step_increment(step)
            adj_max.set_step_increment(step)
            digits = max(0, -order_of_magnitude)
            btn_min.set_digits(digits)
            btn_max.set_digits(digits)

    @property
    def metric_range(self):
        metricMax = self.builder.get_object('MetricMaxAdjustment')
        metricMin = self.builder.get_object('MetricMinAdjustment')
        return NormRange(metricMin.get_value(), metricMax.get_value())

    @metric_range.setter
    def metric_range(self, value):
        adj = (self.builder.get_object('MetricMinAdjustment'),
               self.builder.get_object('MetricMaxAdjustment'))
        btn = (self.builder.get_object('MetricMin'),
               self.builder.get_object('MetricMax'))
        self._set_adjustments(adjustments=adj, value_range=value,
                              spin_buttons=btn)

    @property
    def alpha_range(self):
        alphaMax = self.builder.get_object('AlphaMaxAdjustment')
        alphaMin = self.builder.get_object('AlphaMinAdjustment')
        return NormRange(alphaMin.get_value(), alphaMax.get_value())

    @alpha_range.setter
    def alpha_range(self, value):
        adj = (self.builder.get_object('AlphaMinAdjustment'),
               self.builder.get_object('AlphaMaxAdjustment'))
        btn = (self.builder.get_object('AlphaMin'),
               self.builder.get_object('AlphaMax'))
        self._set_adjustments(adjustments=adj, value_range=value,
                              spin_buttons=btn)

    @property
    def show_alpha(self):
        state = self.builder.get_object('AlphaSwitch').get_state()
        return state

    def reset_metric_range(self):
        """Look at the ranges for metric values and automatically determine
        the normalization range, effectively resetting it.
        """
        metrics = self.parent_map.metric(param=self.active_metric)
        self.metric_range = NormRange(min=metrics.min(),
                                      max=metrics.max())

    def reset_alpha_range(self):
        """Look at the ranges for alpha values and automatically determine the
        normalization range, effectively resetting it.
        """
        alphas = self.parent_map.metric(param=self.active_alpha)
        self.alpha_range = NormRange(min=alphas.min(),
                                     max=alphas.max())

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
        filename_label = self.builder.get_object("FilenameLabel")
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
            # Set miscellaneous data
            with self.parent_map.store() as store:
                filename = store.file_basenames[locus].decode()
            filename_label.set_text(filename)
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
            filename_label.set_text(default_text)
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
