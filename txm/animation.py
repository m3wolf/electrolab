from matplotlib import animation


class FrameAnimation(animation.ArtistAnimation):
    def __init__(self, fig, artists, *args, **kwargs):
        self.fig = fig
        self.artists = artists
        self._framedata = artists
        ret = super().__init__(*args, fig=fig, artists=artists, **kwargs)
        return ret

    @property
    def artists(self):
        return self._framedata

    @artists.setter
    def artists(self, value):
        self._framedata = value

    def _step(self, current_idx):
        artists = self._framedata[current_idx]
        self._draw_next_frame(artists, self._blit)
        return True

    def stop(self):
        return self._stop()
