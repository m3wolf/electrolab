"""Descriptions of X-ray energy absorption edge."""

from pandas import Series

from peakfitting import Peak


class KEdge():
    """An X-ray absorption edge. It is defined by a series of energy
    ranges. All energies are assumed to be in units of electron-volts.

    Arguments
    ---------
    *regions: 3-tuples - All the energy regions. Each tuple is of the
        form (start, end, step) and is inclusive at both ends.

    name: string - A human-readable name for this edge (eg "Ni K-edge")

    pre_edge: 2-tuple (start, stop) - Energy range that defines points
        below the edge region, inclusive.

    post_edge: 2-tuple (start, stop) - Energy range that defines points
        above the edge region, inclusive.

    map_range: 2-tuple (start, stop) - Energy range used for
        normalizing maps. If not supplied, will be determine from pre- and
        post-edge arguments.
    """
    regions = []
    pre_edge = None
    post_edge = None
    map_range = None
    _preedge_slope = None
    _preedge_intercept = None

    def all_energies(self):
        energies = []
        for region in self.regions:
            energies += range(region[0], region[1] + region[2], region[2])
        return sorted(list(set(energies)))

    def energies_in_range(self, norm_range=None):
        if norm_range is None:
            norm_range = (self.map_range[0],
                          self.map_range[1])
        energies = [e for e in self.all_energies()
                    if norm_range[0] <= e <= norm_range[1]]
        return energies

    def fit(self, data: Series, width: int=4):
        """Regression fitting. First the pre-edge is linearlized and the
        extended edge normalized. Pending: a step function is fit to
        the edge itself and any gaussian peaks are then added as
        necessary.

        Arguments
        ---------
        data - The X-ray absorbance data. Should be similar to a pandas
          Series. Assumes that the index is energy. This can be a Series of
          numpy arrays, which allows calculation of image frames, etc.
          Returns a tuple of (peak, goodness) where peak is a fitted peak
          object and goodness is a measure of the goodness of fit.

        width - How many points on either side of the maximum to
          fit.
        """
        max_idx = data.index.get_loc(data.argmax())
        left = max_idx - width
        if left < 0:
            left = 0
        right = max_idx + width + 1
        if right > len(data):
            right = len(data)
        subset = data.iloc[left:right]
        # Correct for background
        vertical_offset = subset.min()
        normalized = subset - vertical_offset
        # Perform fitting
        peak = Peak(method="gaussian")
        peak.vertical_offset = vertical_offset
        peak.fit(data=normalized)
        # Save residuals
        goodness = peak.goodness(subset)
        return (peak, goodness)

    def plot(self, ax=None):
        """Plot this edge on an axes. If the edge has been fit to data, then
        this fit will be plotted. Otherwise, just the ranges of the
        edge will be shown.
        """
        print("TODO: Finish edge.plot method")


class NickelKEdge(KEdge):
    regions = [
        (8250, 8310, 20),
        (8324, 8344, 2),
        (8344, 8356, 1),
        (8356, 8360, 2),
        (8360, 8400, 4),
        (8400, 8440, 8),
        (8440, 8640, 50),
    ]
    pre_edge = (8250, 8325)
    post_edge = (8360, 8640)
    map_range = (8341, 8358)

# Dictionaries make it more intuitive to access these edges by element
k_edges = {
    'Ni': NickelKEdge,
}
