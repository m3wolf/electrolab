import os

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
from matplotlib import figure
import numpy as np

class GtkTxmViewer():
    """View a XANES frameset using a Gtk GUI."""
    current_idx = 0
    def __init__(self, frameset):
        self.frameset = frameset
        self.builder = Gtk.Builder()
        # Load the GUI from a glade file
        gladefile = os.path.join(os.path.dirname(__file__), "xanes_viewer.glade")
        self.builder.add_from_file(gladefile)
        self.window = self.builder.get_object('XanesViewerWindow')
        self.window.set_default_size(1000, 1000)
        # Put the non-glade things in the window
        self.create_axes()
        self.update_window()
        # Set event handlers
        handlers = {
            'gtk-quit': Gtk.main_quit,
            'previous-frame': self.previous_frame,
            'next-frame': self.next_frame
        }
        self.builder.connect_signals(handlers)
        # self.window.connect('delete-event', Gtk.main_quit)

    def previous_frame(self, widget):
        self.current_idx = (self.current_idx - 1) % len(self.frameset)
        self.update_window()

    def next_frame(self, widget):
        self.current_idx = (self.current_idx + 1) % len(self.frameset)
        self.update_window()

    def create_axes(self):
        fig = figure.Figure(figsize=(13.8, 10))
        canvas = FigureCanvas(fig)
        canvas.set_size_request(400,400)
        sw = self.builder.get_object("CanvasWindow")
        sw.add(canvas)
        # Draw a test plot
        self.image_ax = fig.gca()

    def update_window(self):
        # Set labels on the sidepanel
        energy_label = self.builder.get_object('EnergyLabel')
        energy_label.set_text(str(self.current_frame().energy))
        background_lab = self.builder.get_object('BackgroundLabel')
        # Re-draw the current image
        self.image_ax.clear()
        self.current_frame().plot_image(ax=self.image_ax)
        self.image_ax.figure.canvas.draw()

    def show(self):
        self.window.show_all()
        Gtk.main()

    def current_frame(self):
        return self.frameset[self.current_idx]
