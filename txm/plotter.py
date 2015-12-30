import gc

from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg
from matplotlib import figure, pyplot

import plots
from .animation import FrameAnimation

class GtkFramesetPlotter():
    """A class that handles the graphic display of TXM data. It should be
    thought of as an interface to a plotting library, such as
    matplotlib."""
    show_particles = False
    xanes_scatter = None
    def __init__(self, frameset):
        self.frameset = frameset
        # For drawing images
        self._fig = figure.Figure(figsize=(13.8, 10))
        self.canvas = FigureCanvasGTK3Agg(self._fig)
        self.canvas.set_size_request(400,400)

    def create_axes(self):
        # Create figure grid layout
        self.image_ax = self._fig.add_subplot(1, 2, 1)
        plots.set_outside_ticks(self.image_ax)
        self.xanes_ax = self._fig.add_subplot(1, 2, 2)

    def draw(self):
        self._fig.canvas.draw()
        self.image_ax.figure.canvas.draw()
        self.xanes_ax.figure.canvas.draw()

    @property
    def figure(self):
        return self._fig

    def plot_xanes_spectrum(self):
        self.xanes_ax.clear()
        self.xanes_scatter = self.frameset.plot_xanes_spectrum(ax=self.xanes_ax)
        self.xanes_ax.figure.canvas.draw()

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
                                            norm=self.frameset.normalizer(),
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
        self.frame_animation = FrameAnimation(fig=self.figure,
                                              artists=[],
                                              event_source=event_source,
                                              blit=True)
        self.refresh_artists()

    def destroy(self):
        """Remove figures and attempt to reclaim memory."""
        self.image_ax.clear()
        self.image_ax.figure.clf()
        pyplot.close(self.image_ax.figure)
        if hasattr(self, 'map_ax'):
            self.map_ax.clear()
            self.map_ax.figure.clf()
            pyplot.close(self.map_ax.figure)
        gc.collect()
