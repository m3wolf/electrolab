import os

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
from matplotlib import figure
import numpy as np

class GtkTxmViewer():
    """View a XANES frameset using a Gtk GUI."""
    def __init__(self):
        self.builder = Gtk.Builder()
        # Load the GUI from a glade file
        gladefile = os.path.join(os.path.dirname(__file__), "xanes_viewer.glade")
        self.builder.add_from_file(gladefile)
        self.window = self.builder.get_object('MainWindow')
        # Put the non-glade things in the window
        self.create_axes()
        # Make sure it exits cleanly
        self.window.connect('delete-event', Gtk.main_quit)

    def create_axes(self):
        fig = figure.Figure(figsize=(13.8, 10))
        canvas = FigureCanvas(fig)
        canvas.set_size_request(400,400)
        sw = self.builder.get_object("CanvasWindow")
        sw.add(canvas)
        # Draw a test plot
        ax = fig.gca()
        t = np.arange(0.0, 1.0, 0.01)
        ax.plot(t, np.sin(2*np.pi*t))

    def show(self):
        self.window.show_all()
        Gtk.main()
