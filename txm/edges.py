from collections import namedtuple

class Edge():
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
    """
    def __init__(self, *regions, name, pre_edge, post_edge):
        self.regions = regions
        self.pre_edge = pre_edge
        self.post_edge = post_edge

    def energies(self):
        energies = []
        for region in self.regions:
            energies += range(region[0], region[1]+region[2], region[2])
        return sorted(list(set(energies)))

k_edges = {
    'Ni': Edge(
        (8250, 8310, 20),
        (8324, 8344, 2),
        (8344, 8356, 1),
        (8356, 8360, 2),
        (8360, 8400, 4),
        (8400, 8440, 8),
        (8440, 8640, 50),
        name="Ni K-edge",
        pre_edge=(8250, 8325),
        post_edge=(8360, 8640)
    ),
}
