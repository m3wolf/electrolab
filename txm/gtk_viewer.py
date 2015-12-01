import os

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GObject
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
from matplotlib import figure
import numpy as np

class GtkTxmViewer():
    play_mode = False
    show_particles = False
    """View a XANES frameset using a Gtk GUI."""
    def __init__(self, frameset):
        self.frameset = frameset
        self.normalizer = frameset.normalizer()
        self.builder = Gtk.Builder()
        # Load the GUI from a glade file
        gladefile = os.path.join(os.path.dirname(__file__), "xanes_viewer.glade")
        self.builder.add_from_file(gladefile)
        self.window = self.builder.get_object('XanesViewerWindow')
        self.window.set_default_size(1000, 1000)
        # Put the non-glade things in the window
        self.create_axes()
        # Set some values
        slider = self.builder.get_object('FrameSlider')
        self.current_adj = self.builder.get_object('CurrentFrame')
        self.current_adj.set_property('upper', len(self.frameset))
        # Set event handlers
        handlers = {
            'gtk-quit': Gtk.main_quit,
            'previous-frame': self.previous_frame,
            'next-frame': self.next_frame,
            'play-frames': self.play_frames,
            'last-frame': self.last_frame,
            'first-frame': self.first_frame,
            'key-release': self.key_pressed,
            'toggle-particles': self.toggle_particles,
            'update-window': self.update_window
        }
        self.builder.connect_signals(handlers)
        # self.image = self.current_frame().plot_image(ax=self.image_ax,  animated=True)
        self.update_window()
        # self.window.connect('delete-event', Gtk.main_quit)

    @property
    def current_idx(self):
        value = self.current_adj.get_property('value')
        return int(value)

    @current_idx.setter
    def current_idx(self, value):
        self.current_adj.set_property('value', value)

    def key_pressed(self, widget, event):
        if event.keyval == Gdk.KEY_Left:
            self.previous_frame()
        elif event.keyval == Gdk.KEY_Right:
            self.next_frame()

    def toggle_particles(self, widget):
        self.show_particles = not self.show_particles
        self.update_window()

    def play_frames(self, widget):
        self.play_mode = widget.get_property('active')
        GObject.timeout_add(0, self.next_frame, None)

    def first_frame(self, widget):
        self.current_idx = 0
        self.update_window()

    def last_frame(self, widget):
        self.current_idx = len(self.frameset) - 1
        self.update_window()

    def previous_frame(self, widget=None):
        self.current_idx = (self.current_idx - 1) % len(self.frameset)
        self.update_window()

    def next_frame(self, widget=None):
        self.current_idx = (self.current_idx + 1) % len(self.frameset)
        self.update_window()
        if self.play_mode:
            return True
        else:
            return False

    def create_axes(self):
        fig = figure.Figure(figsize=(13.8, 10))
        canvas = FigureCanvas(fig)
        canvas.set_size_request(400,400)
        sw = self.builder.get_object("CanvasWindow")
        sw.add(canvas)
        # Draw a test plot
        self.image_ax = fig.gca()

    def update_window(self, widget=None):
        current_frame = self.current_frame()
        # Set labels on the sidepanel
        energy_label = self.builder.get_object('EnergyLabel')
        energy_label.set_text(str(current_frame.energy))
        x_label = self.builder.get_object('XPosLabel')
        x_label.set_text(str(current_frame.sample_position.x))
        y_label = self.builder.get_object('YPosLabel')
        y_label.set_text(str(current_frame.sample_position.y))
        z_label = self.builder.get_object('ZPosLabel')
        z_label.set_text(str(current_frame.sample_position.z))
        # Re-draw each frame
        self.image_ax.clear()
        self.image_ax.set_aspect(1)
        # self.image.set_data(self.current_image())
        img_ax = self.current_frame().plot_image(ax=self.image_ax,
                                                 norm=self.normalizer,
                                                 show_particles=self.show_particles)
        # if self.show_particles:
        #     self.current_frame().plot_particle_labels(ax=img_ax.axes)
        self.image_ax.figure.canvas.draw()

    def show(self):
        self.window.show_all()
        Gtk.main()

    def current_image(self):
        return self.images[self.current_idx]

    def current_frame(self):
        return self.frameset[self.current_idx]
