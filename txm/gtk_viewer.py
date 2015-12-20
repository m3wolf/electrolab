import os

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GObject
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
from matplotlib import figure, pyplot
import numpy as np

from utilities import xycoord
from txm.frame import Pixel, xy_to_pixel, pixel_to_xy

class GtkTxmViewer():
    play_mode = False
    show_particles = False
    display_type = 'corrected'
    active_pixel = None
    active_xy = None
    map_crosshairs = None
    show_map = True
    show_map_background = True
    apply_edge_jump = False
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
        self.map_window = self.builder.get_object('MapViewerWindow')
        # Put the non-glade things in the window
        self.create_axes()
        # Set some initial values
        slider = self.builder.get_object('FrameSlider')
        self.current_adj = self.builder.get_object('CurrentFrame')
        self.current_adj.set_property('upper', len(self.frameset)-1)
        self.xanes_spectrum = frameset.xanes_spectrum()
        # Populate the combobox with list of available HDF groups
        self.group_combo = self.builder.get_object('ActiveGroupCombo')
        self.group_list = Gtk.ListStore(str, str)
        for group in self.frameset.hdf_group().keys():
            uppercase = " ".join([word.capitalize() for word in group.split('_')])
            tree_iter = self.group_list.append([uppercase, group])
            # Save active group for later initialization
            if group == self.frameset.active_groupname:
                self.active_group = tree_iter
        # Add background frames as an option
        self.group_list.append(['Background Frames', 'background_frames'])
        self.group_combo.set_model(self.group_list)
        # Disable the "show map" button if map is not calculated
        if not self.frameset.map_name:
            btn = self.builder.get_object("ShowMapButton")
            btn.set_sensitive(False)
        # Set initial active group name
        self.group_combo.set_active_iter(self.active_group)
        self.active_groupname = self.frameset.active_groupname
        # Set event handlers
        handlers = {
            'gtk-quit': Gtk.main_quit,
            'previous-frame': self.previous_frame,
            'next-frame': self.next_frame,
            'play-frames': self.play_frames,
            'last-frame': self.last_frame,
            'first-frame': self.first_frame,
            'key-release-main': self.key_pressed_main,
            'key-release-map': self.navigate_map,
            'toggle-particles': self.toggle_particles,
            'update-window': self.update_window,
            'change-active-group': self.change_active_group,
            'launch-map-window': self.launch_map_window,
            'hide-map-window': self.hide_map_window,
            'toggle-map': self.toggle_map_visible,
            'toggle-map-background': self.toggle_map_background,
            'toggle-edge-jump': self.toggle_edge_jump,
        }
        self.builder.connect_signals(handlers)
        # self.image = self.current_frame().plot_image(ax=self.image_ax,  animated=True)
        self.update_window()
        self.window.connect('delete-event', self.quit)

    def toggle_map_visible(self, widget, object=None):
        self.show_map = widget.get_active()
        self.update_map_window()

    def toggle_edge_jump(self, widget, object=None):
        self.apply_edge_jump = widget.get_active()
        self.draw_map()
        self.update_map_window()

    def toggle_map_background(self, widget, object=None):
        self.show_map_background = widget.get_active()
        self.update_map_window()

    def quit(self, widget, object=None):
        self.map_window.destroy()
        Gtk.main_quit()

    def draw_map(self):
        figure = self.map_ax.figure
        figure.clear()
        self.map_ax = figure.gca()
        self.map_crosshairs = None
        # Plot the absorbance background image
        self.bg_artist = self.frameset.plot_mean_image(ax=self.map_ax)
        #     alpha = 0.4
        # else:
        #     alpha = 1
        # if self.apply_edge_jump:
        # else:
        #     # Plot the overall map
        self.map_artist = self.frameset.plot_map(
            ax=self.map_ax,
            edge_jump_filter=self.apply_edge_jump,
            return_type="artist")
        self.map_ax.figure.canvas.draw()

    def hide_map_window(self, widget, object=None):
        print(widget)
        self.map_window.hide()
        return True

    def launch_map_window(self, widget):
        if not hasattr(self, 'map_ax'):
            # Create map axes objects
            map_fig = figure.Figure(figsize=(13.8, 10))
            canvas = FigureCanvas(map_fig)
            canvas.set_size_request(400, 400)
            map_sw = self.builder.get_object("MapWindow")
            map_sw.add(canvas)
            self.map_ax = map_fig.gca()
            # Plot the overall map
            self.draw_map()
            # Connect handlers for clicking on a pixel
            map_fig.canvas.mpl_connect('button_press_event', self.click_map_pixel)
            # Create Xanes axes object
            fig = figure.Figure(figsize=(13.8, 10))
            canvas = FigureCanvas(fig)
            canvas.set_size_request(400, 400)
            xanes_sw = self.builder.get_object("XanesMapWindow")
            xanes_sw.add(canvas)
            self.map_detail_ax = fig.gca()
            self.plot_map_xanes()
            # Set initial state for background switch
            switch = self.builder.get_object('BackgroundSwitch')
            switch.set_active(self.show_map_background)
        # Launch the window
        self.map_window.show_all()

    def click_map_pixel(self, event):
        if event.inaxes == self.map_ax:
            # Convert xy position to pixel values
            xy = xycoord(x=event.xdata, y=event.ydata)
            pixel = xy_to_pixel(xy, extent=self.frameset.extent(),
                                shape=self.frameset.map_shape())
            self.active_pixel = pixel
            self.active_xy = xy
        else:
            self.active_pixel = None
            self.active_xy = None
        self.update_map_window()

    def update_map_window(self):
        self.plot_map_xanes()
        # Show position of active pixel
        if self.active_pixel is None:
            s = "None"
        else:
            s = "(H:{h}, V:{v})".format(h=self.active_pixel.horizontal,
                                        v=self.active_pixel.vertical)
        label = self.builder.get_object("ActivePixelLabel")
        label.set_text(s)
        # Remove old cross-hairs
        if self.map_crosshairs:
            for line in self.map_crosshairs:
                line.remove()
            self.map_crosshairs = None
        # Draw cross-hairs on the map if there's an active pixel
        if self.active_pixel:
            if self.show_map_background:
                color = "white"
            else:
                color = "black"
            xline = self.map_ax.axvline(x=self.active_xy.x,
                                        color=color, linestyle="--")
            yline = self.map_ax.axhline(y=self.active_xy.y,
                                        color=color, linestyle="--")
            self.map_crosshairs = (xline, yline)
        # Show or hide maps as dictated by GUI toggle buttons
        # def remove(ax, artist):
        #     try:
        #         ax.remove(artist)
        #     except ValueError:
        #         pass
        if self.show_map_background:
            self.bg_artist.set_alpha(1)
            # self.map_ax.add_image(self.bg_artist)
            map_alpha = 0.4
        else:
            self.bg_artist.set_alpha(0)
            map_alpha = 1
        if self.show_map:
            self.map_artist.set_alpha(map_alpha)
        else:
            self.map_artist.set_alpha(0)
        # Force redraw in case GTK doesn't have to update
        self.map_ax.figure.canvas.draw()
        self.map_detail_ax.figure.canvas.draw()

    def plot_map_xanes(self):
        self.map_detail_ax.clear()
        self.frameset.plot_xanes_spectrum(ax=self.map_detail_ax,
                                          pixel=self.active_pixel)

    def navigate_map(self, widget, event):
        """Navigate around the map using keyboard."""
        if self.active_pixel is not None:
            horizontal = self.active_pixel.horizontal
            vertical = self.active_pixel.vertical
            if event.keyval == Gdk.KEY_Left:
                horizontal = horizontal - 1
            elif event.keyval == Gdk.KEY_Right:
                horizontal = horizontal + 1
            elif event.keyval == Gdk.KEY_Up:
                vertical = vertical - 1
            elif event.keyval == Gdk.KEY_Down:
                vertical = vertical + 1
            self.active_pixel = Pixel(horizontal=horizontal, vertical=vertical)
            self.active_xy = pixel_to_xy(self.active_pixel,
                                         extent=self.frameset.extent(),
                                         shape=self.frameset.map_shape())
        self.update_map_window()

    def change_active_group(self, widget, object=None):
        """Update to a new frameset HDF group after user has changed combobox."""
        new_group = self.group_list[widget.get_active_iter()][1]
        self.active_groupname = new_group
        if not new_group == 'background_frames':
            self.frameset.switch_group(new_group)
        # Save new xanes spectrum
        self.xanes_spectrum = self.frameset.xanes_spectrum()
        self.draw_xanes_spectrum()
        # Re-normalize for new frameset and display new set to user
        if new_group == 'background_frames':
            self.normalizer = self.frameset.background_normalizer()
        else:
            self.normalizer = self.frameset.normalizer()
        self.update_window()

    @property
    def current_idx(self):
        value = self.current_adj.get_property('value')
        return int(value)

    @current_idx.setter
    def current_idx(self, value):
        self.current_adj.set_property('value', value)

    def key_pressed_main(self, widget, event):
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
        # For drawing images
        fig = figure.Figure(figsize=(13.8, 10))
        canvas = FigureCanvas(fig)
        canvas.set_size_request(400,400)
        image_window = self.builder.get_object("ImageWindow")
        image_window.add(canvas)
        self.image_ax = fig.gca()
        # For drawing XANES spectra
        fig = figure.Figure(figsize=(6, 4))
        canvas = FigureCanvas(fig)
        canvas.set_size_request(300, 300)
        xanes_window = self.builder.get_object("XanesWindow")
        xanes_window.add(canvas)
        self.xanes_ax = fig.gca()
        self.draw_xanes_spectrum()

    def draw_xanes_spectrum(self):
        self.xanes_ax.clear()
        self.frameset.plot_xanes_spectrum(ax=self.xanes_ax)

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
        particle_label = self.builder.get_object('ActiveParticleLabel')
        particle_label.set_text(str(current_frame.active_particle_idx))
        shape_label = self.builder.get_object('ShapeLabel')
        shape_label.set_text(str(current_frame.image_data.shape))
        norm_label = self.builder.get_object('NormLabel')
        norm_text = '[{}, {}]'.format(round(self.normalizer.vmin, 2),
                                      round(self.normalizer.vmax, 2))
        norm_label.set_text(norm_text)
        # Re-draw each frame
        self.image_ax.clear()
        self.image_ax.set_aspect(1)
        # Determine what type of data to present
        key = self.current_frame().image_data.name.split('/')[-1]
        norm = self.normalizer
        if self.active_groupname == 'background_frames':
            data = self.frameset.hdf_file()[self.frameset.background_groupname][key]
        else:
            data = None
        img_ax = self.current_frame().plot_image(data = data,
                                                 ax=self.image_ax,
                                                 norm=norm,
                                                 show_particles=self.show_particles)
        self.image_ax.figure.canvas.draw()
        # Remove old highlighted point from Xanes spectrum
        previous_highlight = getattr(self, 'xanes_highlight', None)
        if previous_highlight:
            try:
                previous_highlight[0].remove()
            except ValueError:
                pass
        # Update the highlighted point on the Xanes spectrum plot
        energy = current_frame.energy
        intensity = self.xanes_spectrum[energy]
        self.xanes_highlight = self.xanes_ax.plot([energy], [intensity], 'ro')
        self.xanes_ax.figure.canvas.draw()

    def show(self):
        self.window.show_all()
        Gtk.main()
        Gtk.main_quit()

    def current_image(self):
        return self.images[self.current_idx]

    def current_frame(self):
        return self.frameset[self.current_idx]
