import gc
from collections import namedtuple

from matplotlib import figure, pyplot, cm, animation
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.backends.backend_agg import FigureCanvasAgg
# from matplotlib.backends.backend_gtk import FigureCanvasGTK
import numpy as np
# May not import if not installed
try:
    from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg
except TypeError:
    pass

import plots
from .animation import FrameAnimation

class FramesetPlotter():
    """A class that handles the graphic display of TXM data. It should be
    thought of as an interface to a plotting library, such as
    matplotlib."""
    map_cmap = "plasma"
    def __init__(self, frameset, map_ax=None):
        self.map_ax = map_ax
        self.frameset = frameset

    def draw_colorbar(self, norm_range=None):
        """Add colormap to the side of the axes."""
        norm = self.map_normalizer(norm_range=norm_range)
        energies = self.frameset.edge.energies_in_range(norm_range=norm_range)
        mappable = cm.ScalarMappable(norm=norm, cmap=self.map_cmap)
        mappable.set_array(np.arange(0, 3))
        self.cbar = pyplot.colorbar(mappable,
                                    ax=self.map_ax,
                                    ticks=energies[0:-1],
                                    spacing="proportional")
        self.cbar.ax.xaxis.get_major_formatter().set_useOffset(False)
        self.cbar.ax.set_title('eV')

    def map_normalizer(self, norm_range=None):
        cmap = cm.get_cmap(self.map_cmap)
        energies = self.frameset.edge.energies_in_range(norm_range=norm_range)
        norm = BoundaryNorm(energies, cmap.N)
        return norm

    def draw_map(self, norm_range=None, alpha=1,
                 edge_jump_filter=False, *args, **kwargs):
        """Draw a map on the map_ax. If no axes exist, a new Axes is created
        with a colorbar."""
        # Construct a discrete normalizer so the colorbar is also discrete
        norm = self.map_normalizer(norm_range=norm_range)
        # Create a new axes if necessary
        if self.map_ax is None:
            self.map_ax = plots.new_image_axes()
            self.draw_colorbar() # norm=norm, ticks=energies[0:-1])
        # Plot chemical map (on top of absorbance image, if present)
        extent = self.frameset.extent()
        masked_map = self.frameset.masked_map(edge_jump_filter=edge_jump_filter)
        artist = self.map_ax.imshow(masked_map,
                                    extent=extent,
                                    cmap=self.map_cmap,
                                    norm=norm,
                                    alpha=alpha,
                                    *args, **kwargs)
        # Decorate axes labels, etc
        self.map_ax.set_xlabel("TODO: Adjust extent when zooming and cropping! (µm)")
        self.map_ax.set_ylabel("µm")
        return artist

    def plot_xanes_spectrum(self):
        self.xanes_scatter = self.frameset.plot_xanes_spectrum(ax=self.xanes_ax)

    def set_title(self, title):
        self.map_ax.set_title(title)

    def draw_crosshairs(self, active_xy=None, color="black"):
        # Remove old cross-hairs
        if self.map_crosshairs:
            for line in self.map_crosshairs:
                line.remove()
            self.map_crosshairs = None
        # Draw cross-hairs on the map if there's an active pixel
        if active_xy:
            xline = self.map_ax.axvline(x=active_xy.x,
                                        color=color, linestyle="--")
            yline = self.map_ax.axhline(y=active_xy.y,
                                        color=color, linestyle="--")
            self.map_crosshairs = (xline, yline)
        self.map_ax.figure.canvas.draw()

class FramesetMoviePlotter(FramesetPlotter):
    show_particles = False
    def create_axes(self, figsize=(13.8, 6)):
        # self.figure = pyplot.figure(figsize=(13.8, 6))
        self.figure = figure.Figure(figsize=figsize)
        self.figure.subplots_adjust(bottom=0.2)
        self.canvas = FigureCanvasGTK3Agg(figure=self.figure)
        # self.canvas = FigureCanvasGTK3Agg(figure=self.figure)
        # Create figure grid layout
        self.image_ax = self.figure.add_subplot(1, 2, 1)
        plots.set_outside_ticks(self.image_ax)
        plots.remove_extra_spines(ax=self.image_ax)
        self.xanes_ax = self.figure.add_subplot(1, 2, 2)
        plots.remove_extra_spines(ax=self.xanes_ax)
        return (self.image_ax, self.xanes_ax)

    def connect_animation(self, interval=50, repeat=True, repeat_delay=3000):
        # Draw the non-animated parts of the graphs
        self.plot_xanes_spectrum()
        # Create artists
        all_artists = []
        for frame in self.frameset:
            frame_artist = frame.plot_image(ax=self.image_ax,
                                            show_particles=False,
                                            norm=self.frameset.image_normalizer(),
                                            animated=True)
            # Get Xanes highlight artists
            energy = frame.energy
            intensity = self.frameset.xanes_spectrum()[energy]
            xanes_artists = self.xanes_ax.plot([energy], [intensity], 'ro',
                                               animated=True)
            # xanes_artists.append(xanes_artist[0])
            if self.show_particles:
                # Get particle labels artists
                particle_artists = frame.plot_particle_labels(
                    ax=self.image_ax,
                    extent=frame.extent(),
                    animated=True
                )
            else:
                particle_artists = []
            all_artists.append((frame_artist, *xanes_artists, *particle_artists))
            # all_artists.append((frame_artist,))
        # Prepare animation
        self.frame_animation = animation.ArtistAnimation(fig=self.figure,
                                                         artists=all_artists,
                                                         interval=interval,
                                                         repeat=repeat,
                                                         repeat_delay=repeat_delay,
                                                         blit=True)

    def show(self):
        self.figure.canvas.show()

    def save_movie(self, *args, **kwargs):
        # Set default values
        kwargs['codec'] = kwargs.get('codec', 'h264')
        kwargs['bitrate'] = kwargs.get('bitrate', -1)
        kwargs['writer'] = 'ffmpeg'
        # codec = kwargs.get('codec', 'h264')
        # bitrate = kwargs.pop('bitrate', -1)
        # Generate a writer object
        # if 'writer' not in kwargs.keys():
        #     writer = animation.FFMpegFileWriter(bitrate=bitrate, codec=codec)
        # else:
        #     writer = kwargs['writer']
        self.figure.canvas.draw()
        return self.frame_animation.save(*args, **kwargs)

class GtkFramesetPlotter(FramesetPlotter):
    """Variation of the frameset plotter that uses canvases made for GTK."""
    show_particles = False
    xanes_scatter = None
    map_crosshairs = None
    def __init__(self, frameset):
        super().__init__(frameset=frameset)
        # Figures for drawing images of frames
        self._frame_fig = figure.Figure(figsize=(13.8, 10))
        self.frame_canvas = FigureCanvasGTK3Agg(self._frame_fig)
        self.frame_canvas.set_size_request(400,400)
        # Figures for overall chemical map
        self._map_fig = figure.Figure(figsize=(13.8, 10))
        self.map_canvas = FigureCanvasGTK3Agg(self._map_fig)
        self.map_canvas.set_size_request(400, 400)

    def draw_map(self, show_map=True, edge_jump_filter=True,
                 show_background=False):
        # Clear old mapping data
        self.map_ax.clear()
        self.map_crosshairs = None
        artists = []
        # Show or hide maps as dictated by GUI toggle buttons
        if show_background:
            # Plot the absorbance background image
            bg_artist = self.frameset.plot_edge_jump(ax=self.map_ax)
            bg_artist.set_cmap('gray')
            artists.append(bg_artist)
            map_alpha = 0.4
        else:
            map_alpha = 1
        if show_map:
            # Plot the overall map
            map_artist = super().draw_map(edge_jump_filter=edge_jump_filter,
                                          alpha=map_alpha)
            artists.append(map_artist)
        # Force redraw
        self.map_canvas.draw()
        return artists

    def draw_map_xanes(self, active_pixel=None):
        self.map_xanes_ax.clear()
        self.frameset.plot_xanes_spectrum(ax=self.map_xanes_ax,
                                          pixel=active_pixel)
        self.map_canvas.draw()

    def plot_xanes_spectrum(self):
        self.xanes_ax.clear()
        super().plot_xanes_spectrum()
        self.xanes_ax.figure.canvas.draw()

    def create_axes(self):
        # Create figure grid layout
        self.image_ax = self.frame_figure.add_subplot(1, 2, 1)
        plots.set_outside_ticks(self.image_ax)
        self.xanes_ax = self.frame_figure.add_subplot(1, 2, 2)
        # Create mapping axes
        self.map_ax = self.map_figure.add_subplot(1, 2, 1)
        plots.set_outside_ticks(self.map_ax)
        self.draw_colorbar()
        self.map_xanes_ax = self.map_figure.add_subplot(1, 2, 2)

    def draw(self):
        self.frame_figure.canvas.draw()
        self.image_ax.figure.canvas.draw()
        self.xanes_ax.figure.canvas.draw()

    @property
    def frame_figure(self):
        return self._frame_fig

    @property
    def map_figure(self):
        return self._map_fig

    def refresh_artists(self):
        """Prepare artist objects for each frame and animate them for easy
        transitioning."""
        all_artists = []
        self.plot_xanes_spectrum()
        # self.xanes_ax.figure.canvas.draw()
        # Get image artists
        self.image_ax.clear()
        for frame in self.frameset:
            frame_artist = frame.plot_image(ax=self.image_ax,
                                            show_particles=False,
                                            norm=self.frameset.image_normalizer(),
                                            animated=True)
            frame_artist.set_visible(False)
            # Get Xanes highlight artists
            energy = frame.energy
            intensity = self.frameset.xanes_spectrum()[energy]
            xanes_artists = self.xanes_ax.plot([energy], [intensity], 'ro',
                                               animated=True)
            [a.set_visible(False) for a in xanes_artists]
            # xanes_artists.append(xanes_artist[0])
            if self.show_particles:
            # Get particle labels artists
                particle_artists = frame.plot_particle_labels(
                    ax=self.image_ax,
                    extent=frame.extent(),
                    animated=True
                )
                [a.set_visible(False) for a in particle_artists]
            else:
                particle_artists = []
            all_artists.append((frame_artist, *xanes_artists, *particle_artists))
            self.frame_animation.artists = all_artists
        return all_artists

    def connect_animation(self, event_source):
        if hasattr(self, 'frame_animation'):
            # Disconnect old animation
            self.frame_animation.stop()
        self.frame_animation = FrameAnimation(fig=self.frame_figure,
                                              artists=[],
                                              event_source=event_source,
                                              blit=True)
        self.refresh_artists()
        # Forces the animation to show the first frame
        event_source._on_change()
        self.frame_canvas.draw()

    def destroy(self):
        """Remove figures and attempt to reclaim memory."""
        # Delete animation
        if hasattr(self, 'frame_animation'):
            self.frame_animation.stop()
            del self.frame_animation
        # Clear axes and figures
        self.image_ax.clear()
        self.image_ax.figure.clf()
        pyplot.close(self.image_ax.figure)
        if hasattr(self, 'map_ax'):
            self.map_ax.clear()
            self.map_ax.figure.clf()
            pyplot.close(self.map_ax.figure)
        gc.collect()


class DummyGtkPlotter(GtkFramesetPlotter):
    def connect_animation(self, *args, **kwargs):
        pass
