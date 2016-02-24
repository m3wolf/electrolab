import os

import gi
from gi.repository import Gtk, Gdk, GObject

from utilities import xycoord
from .frame import Pixel, xy_to_pixel, pixel_to_xy
from .plotter import GtkFramesetPlotter


gi.require_version('Gtk', '3.0')


class GtkTxmViewer():
    play_mode = False
    display_type = 'corrected'
    active_pixel = None
    active_xy = None
    _current_idx = 0
    animation_delay = 1000 / 15
    show_map = True
    show_map_background = True
    apply_edge_jump = False
    """View a XANES frameset using a Gtk GUI."""
    def __init__(self, frameset, plotter=None):
        if plotter is None:
            plotter = GtkFramesetPlotter(frameset=frameset)
        self.plotter = plotter
        self.plotter.create_axes()
        self.frameset = frameset
        self.builder = Gtk.Builder()
        # Load the GUI from a glade file
        gladefile = os.path.join(os.path.dirname(__file__),
                                 "xanes_viewer.glade")
        self.builder.add_from_file(gladefile)
        self.window = self.builder.get_object('XanesViewerWindow')
        self.image_sw = self.builder.get_object('ImageWindow')
        self.image_sw.add(self.plotter.frame_canvas)
        # Prepare the map window for later
        self.map_sw = self.builder.get_object("MapWindow")
        self.map_sw.add(self.plotter.map_canvas)
        self.map_window = self.builder.get_object('MapViewerWindow')
        self.map_window.maximize()
        # Set initial state for background switch
        switch = self.builder.get_object('BackgroundSwitch')
        switch.set_active(self.show_map_background)
        # Set some initial values
        self.current_adj = self.builder.get_object('CurrentFrame')
        self.current_adj.set_property('upper', len(self.frameset) - 1)
        # Put the non-glade things in the window
        self.plotter.plot_xanes_spectrum()
        # Populate the combobox with list of available HDF groups
        self.group_combo = self.builder.get_object('ActiveGroupCombo')
        self.group_list = Gtk.ListStore(str, str)
        for group in self.frameset.hdf_group().keys():
            uppercase = " ".join(
                [word.capitalize() for word in group.split('_')]
            )
            tree_iter = self.group_list.append([uppercase, group])
            # Save active group for later initialization
            if group == self.frameset.active_groupname:
                self.active_group = tree_iter
        # Add background frames as an option
        bg_iter = self.group_list.append(
            ['Background Frames', 'background_frames']
        )
        if self.frameset.is_background():
            self.active_group = bg_iter
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
            'create-artists': self.refresh_artists,
            'max-frame': self.max_frame,
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
        self.window.connect('delete-event', self.quit)
        # Connect handlers for clicking on a pixel
        self.plotter.map_figure.canvas.mpl_connect('button_press_event',
                                                   self.click_map_pixel)
        # Connect handler for mousing over the frame image
        self.plotter.frame_figure.canvas.mpl_connect('motion_notify_event',
                                                     self.update_current_location)
        # Prepare animation
        self.event_source = FrameChangeSource(viewer=self)
        # Make everything visible
        self.update_window()
        self.plotter.draw()

    def toggle_map_visible(self, widget, object=None):
        self.show_map = widget.get_active()
        self.draw_map_plots()
        self.update_map_window()

    def toggle_edge_jump(self, widget, object=None):
        self.apply_edge_jump = widget.get_active()
        self.draw_map_plots()
        self.update_map_window()

    def toggle_map_background(self, widget, object=None):
        self.show_map_background = widget.get_active()
        self.draw_map_plots()
        self.update_map_window()

    def quit(self, widget, object=None):
        self.map_window.destroy()
        self.play_mode = False
        self.window.destroy()
        # Reclaim memory
        self.plotter.destroy()
        Gtk.main_quit()

    def hide_map_window(self, widget, object=None):
        self.map_window.hide()
        return True

    def launch_map_window(self, widget):
        self.draw_map_plots()
        self.map_window.show_all()

    def click_map_pixel(self, event):
        if event.inaxes == self.plotter.map_ax:
            # Convert xy position to pixel values
            xy = xycoord(x=event.xdata, y=event.ydata)
            self.active_pixel = xy_to_pixel(xy,
                                            extent=self.frameset.extent(),
                                            shape=self.frameset.map_shape())
            # Make sure active_xy is in the center of the pixel
            self.active_xy = pixel_to_xy(self.active_pixel,
                                         extent=self.frameset.extent(),
                                         shape=self.frameset.map_shape())
        else:
            self.active_pixel = None
            self.active_xy = None
        self.draw_map_plots()
        self.update_map_window()

    def update_current_location(self, event):
        x_label = self.builder.get_object('XCursorLabel')
        y_label = self.builder.get_object('YCursorLabel')
        v_label = self.builder.get_object('VCursorLabel')
        h_label = self.builder.get_object('HCursorLabel')
        if event.inaxes == self.plotter.image_ax:
            # Convert xy position to pixel values
            xy = xycoord(x=round(event.xdata, 1), y=round(event.ydata, 1))
            pixel = xy_to_pixel(xy, extent=self.frameset.extent(),
                                shape=self.frameset.map_shape())
            x_label.set_text(str(xy.x))
            y_label.set_text(str(xy.y))
            v_label.set_text(str(pixel.vertical))
            h_label.set_text(str(pixel.horizontal))
        else:
            # Set all the cursor labels to blank values
            s = "--"
            x_label.set_text(s)
            y_label.set_text(s)
            v_label.set_text(s)
            h_label.set_text(s)

    def draw_map_plots(self):
        self.plotter.draw_map(show_map=self.show_map,
                              goodness_filter=self.apply_edge_jump,
                              show_background=self.show_map_background)
        # Show crosshairs to indicate active pixel
        if self.show_map_background:
            color = 'white'
        else:
            color = 'black'
        self.plotter.draw_crosshairs(active_xy=self.active_xy, color=color)
        self.plotter.draw_map_xanes(active_pixel=self.active_pixel)

    def update_map_window(self):
        # Show position of active pixel
        if self.active_pixel is None:
            s = "None"
        else:
            s = "(H:{h}, V:{v})".format(h=self.active_pixel.horizontal,
                                        v=self.active_pixel.vertical)
        label = self.builder.get_object("ActivePixelLabel")
        label.set_text(s)

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
        self.draw_map_plots()
        self.update_map_window()

    def change_active_group(self, widget, object=None):
        """Update to a new frameset HDF group after user has changed combobox."""
        new_group = self.group_list[widget.get_active_iter()][1]
        self.active_groupname = new_group
        self.frameset.switch_group(new_group)
        self.plotter.draw()
        self.plotter.connect_animation(event_source=self.event_source)
        self.update_window()
        self.refresh_artists()

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
        self.plotter.show_particles = not self.plotter.show_particles
        self.refresh_artists()
        self.update_window()

    def play_frames(self, widget):
        self.play_mode = widget.get_property('active')
        if self.play_mode:
            GObject.timeout_add(self.animation_delay, self.next_frame, None)

    def first_frame(self, widget):
        self.current_idx = 0
        self.update_window()

    def last_frame(self, widget):
        self.current_idx = len(self.frameset) - 1
        self.update_window()

    def previous_frame(self, widget=None):
        """Go to the next frame in the sequence (or wrap around if at the
        end).
        """
        self.current_idx = (self.current_idx - 1) % len(self.frameset)
        self.update_window()

    def max_frame(self, widget=None):
        """Find the frame with the highest intensity and active it"""
        spectrum = self.frameset.xanes_spectrum()
        self.current_idx = spectrum.values.argmax()
        self.update_window()

    def next_frame(self, widget=None):
        self.current_idx = (self.current_idx + 1) % len(self.frameset)
        self.update_window()
        if self.play_mode:
            keep_going = True
        else:
            keep_going = False
        return keep_going

    def remove_artists(self):
        """Remove current artists from the plotting axes."""
        for artist_tuple in self.frame_animation.artists:
            for artist in artist_tuple:
                artist.remove()

    def refresh_artists(self, *args, **kwargs):
        self.plotter.connect_animation(self.event_source)
        # self.plotter.refresh_artists(*args, **kwargs)

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
        norm_text = '[{}, {}]'.format(
            round(self.frameset.image_normalizer().vmin, 2),
            round(self.frameset.image_normalizer().vmax, 2)
        )
        norm_label.set_text(norm_text)

    def progress_modal(self, objs, operation='Working'):
        """
        Display the progress of the current operation via print statements.
        """
        modal = self.builder.get_object("WaitingWindow")
        modal.show_all()
        ctr = 1
        for obj in objs:
            ctr += 1
            yield obj
        modal.hide()

    def show(self):
        self.window.show_all()
        self.plotter.connect_animation(event_source=self.event_source)
        Gtk.main()
        Gtk.main_quit()

    def current_image(self):
        return self.images[self.current_idx]

    def current_frame(self):
        return self.frameset[self.current_idx]


class FrameChangeSource():
    callbacks = []

    def __init__(self, viewer):
        self.viewer = viewer

    def add_callback(self, func, *args, **kwargs):
        self.callbacks.append((func, args, kwargs))

    def remove_callback(self, func, *args, **kwargs):
        if args or kwargs:
            self.callbacks.remove((func, args, kwargs))
        else:
            funcs = [c[0] for c in self.callbacks]
            if func in funcs:
                self.callbacks.pop(funcs.index(func))

    def start(self):
        # Listen to the frame adjustment signals
        if not hasattr(self, 'handler_id'):
            self.handler_id = self.viewer.current_adj.connect('value-changed', self._on_change)

    def stop(self):
        if hasattr(self, 'handler_id'):
            self.viewer.current_adj.disconnect(self.handler_id)
            del self.handler_id

    def _on_change(self, widget=None, object=None):
        for func, args, kwargs in self.callbacks:
            func(self.viewer.current_idx, *args, **kwargs)
