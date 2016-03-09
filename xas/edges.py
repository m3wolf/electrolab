"""Descriptions of X-ray energy absorption edge."""

import numpy as np
from pandas import Series
from sklearn import linear_model, svm

from peakfitting import Peak
import plots


class KEdge():
    """An X-ray absorption edge. It is defined by a series of energy
    ranges. All energies are assumed to be in units of electron-volts.

    Attributes
    ---------
    E_0: number - The energy of the absorption edge itself.

    *regions: 3-tuples - All the energy regions. Each tuple is of the
      form (start, end, step) and is inclusive at both ends.

    name: string - A human-readable name for this edge (eg "Ni K-edge")

    pre_edge: 2-tuple (start, stop) - Energy range that defines points
      below the edge region, inclusive.

    post_edge: 2-tuple (start, stop) - Energy range that defines points
      above the edge region, inclusive.

    post_edge_order - What degree polynomial to use for fitting
      the post_edge region.

    map_range: 2-tuple (start, stop) - Energy range used for
      normalizing maps. If not supplied, will be determine from pre-
      and post-edge arguments.
    """
    regions = []
    pre_edge = None
    post_edge = None
    map_range = None
    post_edge_order = 2
    pre_edge_fit = None

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

    def _post_edge_xs(self, x):
        """Convert a set of x values to a power series up to an order
        determined by self.post_edge_order."""
        X = []
        for power in range(1, self.post_edge_order+1):
            X.append(x**power)
        X = np.array(X)
        X = X.swapaxes(0, 1)
        return X

    def fit(self, data: Series, width: int=4):
        """Regression fitting. First the pre-edge is linearlized and the
        extended (post) edge normalized using a polynomial. Pending: a
        step function is fit to the edge itself and any gaussian peaks
        are then added as necessary. This method is taken mostly from
        the Athena manual chapter 4.

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
        # Determine linear background region in pre-edge
        pre_edge = data.ix[self.pre_edge[0]:self.pre_edge[1]]
        self._pre_edge_fit = linear_model.LinearRegression()
        self._pre_edge_fit.fit(
            X=np.array(pre_edge.index).reshape(-1, 1),
            y=pre_edge.values
        )
        # Correct the post edge region with polynomial fit
        post_edge = data.ix[self.post_edge[0]:self.post_edge[1]]
        self._post_edge_fit = linear_model.LinearRegression()
        x = np.array(post_edge.index)
        self._post_edge_fit.fit(
            X=self._post_edge_xs(x),
            y=post_edge.values
        )
        # max_idx = data.index.get_loc(data.argmax())
        # left = max_idx - width
        # if left < 0:
        #     left = 0
        # right = max_idx + width + 1
        # if right > len(data):
        #     right = len(data)
        # subset = data.iloc[left:right]
        # # Correct for background
        # vertical_offset = subset.min()
        # normalized = subset - vertical_offset
        # # Perform fitting
        # peak = Peak(method="gaussian")
        # peak.vertical_offset = vertical_offset
        # peak.fit(data=normalized)
        # # Save residuals
        # goodness = peak.goodness(subset)
        # return (peak, goodness)

    def normalize(self, spectrum: Series) -> Series:
        """Adjust the given spectrum so that the pre-edge is around 0 and the
        post-edge is around 1. The `fit()` method should have been
        previously called, ideally (though not required) on the same data.
        """
        # Calculate predicted pre-edge
        energies = np.array(spectrum.index)
        preedge = self._pre_edge_fit.predict(energies.reshape(-1, 1))
        # Calculate predicted absorbance at whiteline
        abs_0 = self._post_edge_fit.predict(self.E_0)
        abs_0 = abs_0 - self._pre_edge_fit.predict(self.E_0)
        # Perform normalization
        new_spectrum = (spectrum - preedge) / abs_0
        return new_spectrum

    def plot(self, ax=None):
        """Plot this edge on an axes. If the edge has been fit to data, then
        this fit will be plotted. Otherwise, just the ranges of the
        edge will be shown.
        """
        if ax is None:
            ax = plots.new_axes()
        # Find range of values to plot based on edge energies
        all_energies = self.all_energies()
        xmin = min(all_energies)
        xmax = max(all_energies)
        x = np.linspace(xmin, xmax, num=50)
        # Plot pre-edge line
        y = self._pre_edge_fit.predict(x.reshape(-1, 1))
        ax.plot(x, y)
        # Plot post-edge curve
        y = self._post_edge_fit.predict(self._post_edge_xs(x))
        ax.plot(x, y)


class NickelKEdge(KEdge):
    E_0 = 8333
    regions = [
        (8250, 8310, 20),
        (8324, 8344, 2),
        (8344, 8356, 1),
        (8356, 8360, 2),
        (8360, 8400, 4),
        (8400, 8440, 8),
        (8440, 8640, 50),
    ]
    # pre_edge = (8250, 8325)
    pre_edge = (8250, 8290)
    # post_edge = (8352, 8640)
    post_edge = (8440, 8640)
    map_range = (8341, 8358)

# Dictionaries make it more intuitive to access these edges by element
k_edges = {
    'Ni': NickelKEdge,
}
